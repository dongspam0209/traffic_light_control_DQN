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
BATCH_SIZE = 32
GAMMA = 0.99
LR = 1e-4
MEMORY_CAPACITY=10000
Q_NETWORK_ITERATION=100 

#action phase definition
PHASE_NS_GREEN = 0  # action_number 0 code 00 -> 북/남 직진
PHASE_NS_YELLOW = 1 
PHASE_NSL_GREEN = 2  # action_number 1 code 01 -> 북/남 좌회전
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action_number 2 code 10 -> 동/서 직진
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action_number 3 code 11 -> 동/서 좌회전
PHASE_EWL_YELLOW = 7


###################################################################
lane=["W_in_0","W_in_1","W_in_2","W_in_3","N_in_0","N_in_1","N_in_2","N_in_3",
    "E_in_0","E_in_1","E_in_2","E_in_3","S_in_0","S_in_1","S_in_2","S_in_3"]

device=torch.device("cuda"if torch.cuda.is_available() else "cpu")
class Simulation:
    def __init__(self,DQN,ReplayMemory,Cargenerator,sumo_cmd,max_steps,num_states,num_actions,green_duration,yellow_duration,green_turn_duration):
        self.policy_net=DQN(num_states,num_actions).to(device)
        self.target_net=DQN(num_states,num_actions).to(device)

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)

        self._ReplayMemory=ReplayMemory
        self._Cargenerator=Cargenerator
        self._sumo_cmd=sumo_cmd
        self._max_steps=max_steps
        self._num_states=num_states
        self._num_actions=num_actions
        self._green_duration=green_duration
        self._yellow_duration=yellow_duration
        self._green_turn_duration=green_turn_duration
        self._queue_length_per_episode=[]
        self.loss_history=[]

        self._previous_total_waiting_time=0
        self._previous_lane_waiting_times = {"E_in": 0, "N_in": 0, "W_in": 0, "S_in": 0}




    def run(self,episode,epsilon):
        self._Cargenerator.generate_car(seed=episode) # car generation
        traci.start(self._sumo_cmd)
        print("Simulating")

        # inits
        self._step=0
        self._waiting_times = {}
        self._sum_queue_length = 0
        self._sum_waiting_time = 0
        old_state = -1
        old_action_number=-1
        
        while self._step < self._max_steps:
            
            self._reward_queue_length=0 # queue length for reward

            ############# get state ##################
            current_state=self._get_state()                  
            ##########################################

            ############# reward & memory push #######
            reward=self._reward()
            if self._step != 0:
                self._ReplayMemory.push(old_state, old_action_number,current_state,reward)
            ###########################################

            df1=pd.DataFrame(current_state[0],index=lane)
            df1.to_csv('./intersection/generate_exist.csv')
            df2=pd.DataFrame(current_state[1],index=lane)
            df2.to_csv('./intersection/generate_velocity.csv')
            df3=pd.DataFrame(current_state[2],index=lane)
            df3.to_csv('./intersection/generate_waiting_time.csv')

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

        

        print("epsilon",round(epsilon,2))
        self.optimize_model()        
        self._queue_length_per_episode.append(self._sum_queue_length)
        
        traci.close()
    
    def _get_state(self):
        state=np.zeros((3,16,100))
        vehicle_list=traci.vehicle.getIDList()
        lane_group=0

        for veh_id in vehicle_list:
            lane_position = traci.vehicle.getLanePosition(veh_id) # car position in lane
            lane_id=traci.vehicle.getLaneID(veh_id) # -> lane
            lane_position=700-lane_position # traffic light ~> max len of road 0~700
            lane_cell= int(lane_position/7) # 7m : hegiht of the cell

            for idx in range(len(lane)):
                if lane_id==lane[idx]:
                    lane_group=idx

            state[0][lane_group][lane_cell]=1 # cell occupied
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
            return random.randint(0, self._num_actions - 1) # random action
        else:
            with torch.no_grad():
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
            queue_length = self._get_queue_length()
            self._sum_queue_length+=queue_length
            self._reward_queue_length+=queue_length
            


    def _collect_waiting_times(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        incoming_roads = ["E_in", "N_in", "W_in", "S_in"]
        
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                self._waiting_times[car_id] = wait_time
            else:
                if car_id in self._waiting_times: # a car that was tracked has cleared the intersection
                    del self._waiting_times[car_id] 

        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time

    def _collect_waiting_times_per_lane(self):
        incoming_roads = ["E_in", "N_in", "W_in", "S_in"]
        lane_waiting_times = {road: [] for road in incoming_roads}
        car_list = traci.vehicle.getIDList()

        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)  

            if road_id in incoming_roads:
                lane_waiting_times[road_id].append(wait_time)

        aggregated_waiting_times = {road: sum(times) for road, times in lane_waiting_times.items()}
        delta_waiting_times = {road: aggregated_waiting_times[road] - self._previous_lane_waiting_times[road] for road in incoming_roads}
        self._previous_lane_waiting_times = aggregated_waiting_times

        return delta_waiting_times
    

    def _get_queue_length(self):
        """
        Retrieve the number of cars with speed = 0 in every incoming lane
        """
        halt_N = traci.edge.getLastStepHaltingNumber("N_in")
        halt_S = traci.edge.getLastStepHaltingNumber("S_in")
        halt_E = traci.edge.getLastStepHaltingNumber("E_in")
        halt_W = traci.edge.getLastStepHaltingNumber("W_in")
        queue_length = halt_N + halt_S + halt_E + halt_W
        
        return queue_length

    def _get_queue_length_per_lane(self):
        halt_N = traci.edge.getLastStepHaltingNumber("N_in")
        halt_S = traci.edge.getLastStepHaltingNumber("S_in")
        halt_E = traci.edge.getLastStepHaltingNumber("E_in")
        halt_W = traci.edge.getLastStepHaltingNumber("W_in")

        queue_lengths_per_lane={
            "N_in": halt_N,
            "S_in": halt_S,
            "E_in": halt_E,
            "W_in": halt_W,
        }
        return queue_lengths_per_lane

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
        
        for state in batch.state:
            print(torch.tensor(state).size())


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
        w_1=10
        w_2=10
        w_3=10

        current_total_waiting_time=self._collect_waiting_times()
        delta_waiting_time=current_total_waiting_time-self._previous_total_waiting_time
        self._previous_total_waiting_time=current_total_waiting_time
        queue_length=self._reward_queue_length

        avg_waiting_time=1000
        avg_queue_length=1000

        each_waiting_time_for_fairness=self._collect_waiting_times_per_lane()
        each_queue_length_for_fairness=self._get_queue_length_per_lane()
        waiting_time_fairness=self.calculate_fairness_index(list(each_waiting_time_for_fairness.values()))
        queue_length_fairness=self.calculate_fairness_index(list(each_queue_length_for_fairness.values()))

        reward=-(w_1*delta_waiting_time/avg_waiting_time+w_2*queue_length/avg_queue_length+w_3*(w_1/(w_1+w_2)*waiting_time_fairness+ w_2/(w_1+w_2)*queue_length_fairness))
        return reward
    

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
    