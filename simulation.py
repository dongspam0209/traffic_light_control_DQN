import traci
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from replay import Experience
###################################################################
# hyper-parameters
BATCH_SIZE = 128
GAMMA = 0.90
LR = 1e-4
MEMORY_CAPACITY=10000
Q_NETWORK_ITERATION=10 

#action phase definition
PHASE_NS_GREEN = 0  # action_number 0 code 00 -> 북/남 직진
PHASE_NS_YELLOW = 1 
PHASE_NSL_GREEN = 2  # action_number 1 code 01 -> 북/남 좌회전
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action_number 2 code 10 -> 동/서 직진
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action_number 3 code 11 -> 동/서 좌회전
PHASE_EWL_YELLOW = 7

lane=["W_in_0","W_in_1","W_in_2","W_in_3","N_in_0","N_in_1","N_in_2","N_in_3",
    "E_in_0","E_in_1","E_in_2","E_in_3","S_in_0","S_in_1","S_in_2","S_in_3"]

###################################################################

device=torch.device("cuda"if torch.cuda.is_available() else "cpu")


class Simulation:
    def __init__(self,DQN,ReplayMemory,Cargenerator,sumo_cmd,max_steps,num_states,num_actions,green_duration,yellow_duration,green_turn_duration):
        self.policy_net=DQN(num_states,num_actions).to(device)
        self.target_net=DQN(num_states,num_actions).to(device)

        self.learn_step_counter = 0
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)

        self._ReplayMemory=ReplayMemory
        self._Cargenerator=Cargenerator
        self._sumo_cmd=sumo_cmd
        self._max_steps=max_steps
        self._num_states=num_states
        self._num_actions=num_actions

        # traffic signal duration
        self._green_duration=green_duration
        self._yellow_duration=yellow_duration
        self._green_turn_duration=green_turn_duration

        # plot variables
        self._queue_length_per_episode=[]

        self._waiting_time_per_episode=[]

        self.reward_per_epsiode=[]
        self.loss_history=[]

        

    def run(self,episode,epsilon):
        self._Cargenerator.generate_car(seed=episode) # car generation
        traci.start(self._sumo_cmd)
        print("Simulating")

        # plot variables
        self.plot_queue_length=0
        self.plot_wait_time=0
        self.plot_reward=0

        # reward variables
        self.queue_len_per_lane=[]
        self._reward_queue_length = 0 # queue length for reward

        self.wait_times_per_lane=[]
        self._reward_wait_time=0

        # inits
        self._step = 0
        old_state = -1
        old_action_number=-1
        old_total_wait=0
        print("현재 에피소드의 큐 길이: ", self.plot_queue_length)
        print("현재 에피소드의 waiting time: ",self.plot_wait_time)

        while self._step < self._max_steps:
            # print("현재 step은 : ", self._step)
            # print("\n###################\n")
            ############# get state ##################
            current_state=self._get_state()
            df1=pd.DataFrame(current_state[0],index=lane)
            df1.to_csv('./intersection/generate_exist.csv')
            df2=pd.DataFrame(current_state[1],index=lane)
            df2.to_csv('./intersection/generate_velocity.csv')
            df3=pd.DataFrame(current_state[2],index=lane)
            df3.to_csv('./intersection/generate_waiting_time.csv')

            ########## waiting time calculate ###############
            df3=df3.transpose()
            wait_time_sum_per_lane_list=[]
            # 각 방향에 대해 집계
            for direction in ['W_in', 'N_in', 'E_in', 'S_in']:
                # W_in_0
                wait_time_sum_per_lane_list.append(df3[direction + '_0'].sum())
                
                # else lanes
                sum_123 = df3[[direction + '_1', direction + '_2', direction + '_3']].sum().sum()
                wait_time_sum_per_lane_list.append(sum_123)

            self.wait_times_per_lane=wait_time_sum_per_lane_list
            current_total_wait=sum(wait_time_sum_per_lane_list)
            self._reward_wait_time=current_total_wait-old_total_wait
            #################################################
            ########## queue_length calculate ###############
            df1=df1.transpose()
            df2=df2.transpose()
            queue_length_sum_per_lane_list=[]
            # 각 방향에 대해 집계
            for direction in ['W_in', 'N_in', 'E_in', 'S_in']:
                # _0 lane - 차량 존재하고, 속도가 0.1보다 작은 경우 count
                halted_count = ((df1[direction + '_0'] == 1) & (df2[direction + '_0'] <= 0.1)).sum()
                queue_length_sum_per_lane_list.append(halted_count)
                
                # else lanes - 차량 존재하고, 속도가 0.1보다 작은 경우 count
                # 각 차선에 대해 조건 적용 후 sum
                halted_count_else = sum(
                    ((df1[direction + '_' + str(i)] == 1) & (df2[direction + '_' + str(i)] <= 0.1)).sum()
                    for i in range(1, 4)
                )
                queue_length_sum_per_lane_list.append(halted_count_else)

            self.queue_len_per_lane=queue_length_sum_per_lane_list
            self._reward_queue_length=sum(queue_length_sum_per_lane_list)
            self.plot_queue_length+=self._reward_queue_length
            #################################################
            ############## to plot waiting time in one episode (total sum of waiting time in whole one episode ###################
            for lane_group in range (16):
                for lane_cell in range (100):
                    self.plot_wait_time += current_state[2][lane_group][lane_cell]

            #######################################################################################################################

            ############# reward & memory push #######
            reward=self._reward()
            self.plot_reward+=reward
            if self._step != 0:
                self._ReplayMemory.push(old_state, old_action_number,current_state,reward)
            ###########################################

            ############ action select ##############
            action_to_do=self._choose_action(current_state,epsilon)
            if self._step != 0 and old_action_number != action_to_do:
                self._set_yellow_phase(old_action_number)
                self._simulate(self._yellow_duration)

            duration=self._set_green_phase(action_to_do)
            self._simulate(duration)
            ###########################################

            old_state=current_state
            old_action_number=action_to_do
            old_total_wait=current_total_wait
            self.optimize_model()        


        self.reward_per_epsiode.append(self.plot_reward/self._step)
        self._waiting_time_per_episode.append(self.plot_wait_time / self._step)
        self._queue_length_per_episode.append(self.plot_queue_length / self._step)

        print("epsilon",round(epsilon,2))
        # print("큐 길이: ", self._queue_length_per_episode)
       
        
        traci.close()
    
    def _get_state(self):
        state=np.zeros((3,16,100))
        vehicle_list=traci.vehicle.getIDList()
        lane_group=0
        # print(len(vehicle_list))
        # print('get state')
        for veh_id in vehicle_list:
            lane_position = traci.vehicle.getLanePosition(veh_id) # car position in lane
            lane_id=traci.vehicle.getLaneID(veh_id) # -> lane
            # print(lane_id)
            lane_position=700-lane_position # traffic light ~> max len of road 0~700
            lane_cell= int(lane_position/7) # 7m : hegiht of the cell
            lane_cell=min(lane_cell,99)

            for idx in range(len(lane)):
                if lane_id==lane[idx]:
                    lane_group=idx

            state[0][lane_group][lane_cell] = 1 # cell occupied
            state[1][lane_group][lane_cell]=traci.vehicle.getSpeed(veh_id) # vehicle velocity
            state[2][lane_group][lane_cell]=traci.vehicle.getAccumulatedWaitingTime(veh_id) # waiting time

        
        
        return state.tolist()

    def _set_yellow_phase(self, prev_action_num):
        """
        Activate the correct yellow light combination in sumo
        """
        yellow_phase_code = prev_action_num * 2 + 1 # obtain the yellow phase code, based on the old action (ref on environment.net.xml)
        traci.trafficlight.setPhase("intersection", yellow_phase_code)

    def _set_green_phase(self, action_number):
        """
        Activate the correct green light combination in sumo
        """
        if action_number == 0:
            traci.trafficlight.setPhase("intersection", PHASE_NS_GREEN)
            return self._green_duration
        elif action_number == 1:
            traci.trafficlight.setPhase("intersection", PHASE_NSL_GREEN)
            return self._green_turn_duration
        elif action_number == 2:
            traci.trafficlight.setPhase("intersection", PHASE_EW_GREEN)
            return self._green_duration
        elif action_number == 3:
            traci.trafficlight.setPhase("intersection", PHASE_EWL_GREEN)
            return self._green_turn_duration

    def _choose_action(self, state, epsilon):
        """
        Decide wheter to perform an explorative or exploitative action, according to an epsilon-greedy policy
        """
        state_tensor=torch.tensor([state],device=device,dtype=torch.float)
        if random.random() < epsilon:
            # print("탐험중임. 현재 값: ", random.random())
            return random.randint(0, self._num_actions - 1) # random action
        else:
            with torch.no_grad():
                # print("탐험 안 함 : ", random.random())
                return self.policy_net(state_tensor).max(1)[1].item() # the best action given the current state    

    def _simulate(self, steps_todo):
        """
        Execute steps in sumo while gathering statistics
        """
        if (self._step + steps_todo) >= self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._step += 1 # update the step counter
            steps_todo -= 1

        

    def optimize_model(self):

        if len(self._ReplayMemory) < BATCH_SIZE:
            return

        experience = self._ReplayMemory.sample(BATCH_SIZE)
        batch = Experience(*zip(*experience))

        # 최종이 아닌 상태의 마스크를 계산하고 배치 요소를 연결합니다
        # (최종 상태는 시뮬레이션이 종료 된 이후의 상태)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).view(-1,3,16,100).to(device)
        # non_final_next_states = torch.cat([torch.tensor(s, dtype=torch.float).unsqueeze(0) for s in batch.next_state if s is not None]).to(device)

        state_batch = torch.cat(batch.state).view(BATCH_SIZE,3,16,100).to(device)
        # state_batch = torch.cat([torch.tensor(s, dtype=torch.float).unsqueeze(0) for s in batch.state]).to(device)
        action_batch = torch.cat(batch.action).view(-1,1).to(device)
        reward_batch = torch.cat(batch.reward).to(device)

        q_eval = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=device)

        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
       
        # 기대 Q 값 계산
        q_target = (next_state_values * GAMMA) + reward_batch

        # MSELoss 손실 계산
        criterion = nn.MSELoss()
        loss = criterion(q_eval, q_target.unsqueeze(1))

        # 모델 최적화
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss_history.append(loss.item())
       
        #update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION ==0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.learn_step_counter+=1

    def _reward(self):
        w_1=1/3
        w_2=1/3
        w_3=1/3

        waiting_time=self._reward_wait_time
        queue_length=self._reward_queue_length

        avg_waiting_time=80
        avg_queue_length=1

        each_waiting_time_for_fairness=self.wait_times_per_lane
        each_queue_length_for_fairness=self.queue_len_per_lane
        waiting_time_fairness=self.calculate_fairness_index(each_waiting_time_for_fairness)
        queue_length_fairness=self.calculate_fairness_index(each_queue_length_for_fairness)

        reward=-(w_1*waiting_time/avg_waiting_time + w_2*queue_length/avg_queue_length+w_3*(w_1/(w_1+w_2)*waiting_time_fairness+ w_2/(w_1+w_2)*queue_length_fairness))
        
        return reward
    
    # fairness index 0~1
    def calculate_fairness_index(self,values):
        if not values:
            return 1.0
        square_of_sums=np.square(np.sum(values))
        sum_of_squares=np.sum(np.square(values))
        if sum_of_squares > 0:
            return square_of_sums/(len(values)*sum_of_squares) 
        else:
            return 1
    

    @property
    def queue_length_store(self):
        return self._queue_length_per_episode
    @property
    def loss_store(self):
        return self.loss_history
    @property
    def wait_time_store(self):
        return self._waiting_time_per_episode
    
    @property
    def reward_store(self):
        return self.reward_per_epsiode
    