import traci
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from replay import Experience


# Hyper-parameters
BATCH_SIZE = 32
GAMMA = 0.90
LR = 1e-4
MEMORY_CAPACITY = 10000
Q_NETWORK_ITERATION = 10

# Action phase definition
PHASE_NS_GREEN = 0  # action_number 0 code 00 -> 북/남 직진
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # action_number 1 code 01 -> 북/남 좌회전
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action_number 2 code 10 -> 동/서 직진
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action_number 3 code 11 -> 동/서 좌회전
PHASE_EWL_YELLOW = 7

lane = ["W_in_0", "W_in_1", "W_in_2", "W_in_3", "N_in_0", "N_in_1", "N_in_2", "N_in_3",
        "E_in_0", "E_in_1", "E_in_2", "E_in_3", "S_in_0", "S_in_1", "S_in_2", "S_in_3"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Simulation:
    def __init__(self, DQN, ReplayMemory, Cargenerator, sumo_cmd, max_steps, num_states, _num_actions, green_duration,
                 green_turn_duration, cycle_duration=40):
        self._cycle_duration = cycle_duration
        self._action_ratios = [
            [7, 7, 7, 7],      # 1:1:1:1
            [6, 6, 6, 10],     # 1:1:1:2
            [6, 6, 10, 6],     # 1:1:2:1
            [5, 5, 9, 9],      # 1:1:2:2
            [6, 10, 6, 6],     # 1:2:1:1
            [5, 9, 5, 9],      # 1:2:1:2
            [5, 9, 9, 5],      # 1:2:2:1
            [4, 8, 8, 8],      # 1:2:2:2
            [10, 6, 6, 6],     # 2:1:1:1
            [9, 5, 5, 9],      # 2:1:1:2
            [9, 5, 9, 5],      # 2:1:2:1
            [8, 4, 8, 8],      # 2:1:2:2
            [9, 9, 5, 5],      # 2:2:1:1
            [8, 8, 4, 8],      # 2:2:1:2
            [8, 8, 8, 4],      # 2:2:2:1
        ]
        self._num_actions = len(self._action_ratios) # 15개의 액션

        self.policy_net = DQN(num_states, self._num_actions).to(device)
        self.target_net = DQN(num_states, self._num_actions).to(device)

        self.learn_step_counter = 0
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)

        self._ReplayMemory = ReplayMemory
        self._Cargenerator = Cargenerator
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._num_states = num_states

        # Traffic signal duration 남북 직진, 남북 좌회전, 동서 직진, 동서 좌회전 순서대로이고, 중간중간에 노란 신호
        self.phases = [PHASE_NS_GREEN, PHASE_NS_YELLOW, PHASE_NSL_GREEN, PHASE_NSL_YELLOW,
                       PHASE_EW_GREEN, PHASE_EW_YELLOW, PHASE_EWL_GREEN, PHASE_EWL_YELLOW]
        self.current_phase_index = 0 # 첫 시작 신호는 남북직진

        # Plot variables 에피소드별로 각각 대기열,대기시간,리워드,손실값(loss) 계산하기 위해 선언
        self._queue_length_per_episode = []
        self._waiting_time_per_episode = []
        self.reward_per_episode = []
        self.loss_history = []

    # run함수는 에피소드마다 실행되는 함수로, 실질적인 시뮬레이션 실행 함수임.
    def run(self, episode, epsilon):

        ### 시뮬레이션 시작 전 차량 생성하는 클래스 호출 및 차량 생성파일 생성 → generator.py 통해서 cross.rou.xml파일 생성함 ###
        if self._Cargenerator is not None:
            self._Cargenerator.generate_car(seed=episode)

        ### traci.start로 시뮬레이션 시작 ###
        traci.start(self._sumo_cmd)
        print("Simulating")

        # Plot variables
        # 에피소드별마다 대기열,대기시간,리워드를 얼마만큼 쌓였는지 모두 계산하기 위해서, 시뮬레이션 시작 초기 단계에 모두 0으로 선언 및 초기화
        self.plot_queue_length = 0
        self.plot_wait_time = 0
        self.plot_reward = 0

        # Reward variables
        # 리워드 변수로 각 차선별 대기열(queue_len_per_lane), 모든 차선의 대기열 합한 값(_reward_queue_length).
        self.queue_len_per_lane = [] 
        self._reward_queue_length = 0  
        # 리워드 변수로 각 차선별 대기시간(wait_times_per_lane), 모든 차선의 대기시간 합한 값(_reward_wait_time).
        self.wait_times_per_lane = []
        self._reward_wait_time = 0

        # Inits
        #step은 시뮬레이션 초의 의미. 0초로 초기화. max_steps는 3600으로 선언함. 즉 3600초동안 시뮬레이션이 돌아감.
        self._step = 0
        #### vehicle duration calculate ####
        self.veh_time_in_lane={}
        self.veh_wait_time_in_lane={}
        #최초의 s_t=-1 a_t=-1 로 초기화, 어차피 맨처음 초기화 한 값은 메모리에 넣지 않을 것이기 때문에, 초기화만 해둠
        old_state = -1
        old_action_number = -1

        previous_cycle_queue_length = 0  # 이전 사이클의 큐 길이
        previous_cycle_wait_time = 0  # 이전 사이클의 대기 시간

        while self._step < self._max_steps:
            ############# Get state ##################
            ## get state로 df1에 차량 존재유무, df2에 차량의 속도, df3에 차량의 대기시간 작성 자세한 내용은 _get_state함수 확인
            current_state = self._get_state()
            df1 = pd.DataFrame(current_state[0], index=lane)
            df1.to_csv('./intersection/generate_exist.csv')
            df2 = pd.DataFrame(current_state[1], index=lane)
            df2.to_csv('./intersection/generate_velocity.csv')
            df3 = pd.DataFrame(current_state[2], index=lane)
            df3.to_csv('./intersection/generate_waiting_time.csv')

            ########## Waiting time calculate ###############
            #대기시간 계산함수
            df3 = df3.transpose()
            wait_time_sum_per_lane_list = []

            # 각 방향에 대해 집계
            for direction in ['W_in', 'N_in', 'E_in', 'S_in']:
                # W_in_0 N_in_0 E_in_0 S_in_0으로 각 진입차도에 해당하는 좌회전 차선에 대한 대기시간 모두 함해서 wait_time_sum_per_lane_list에 넣기.
                wait_time_sum_per_lane_list.append(df3[direction + '_0'].sum())

                # else lanes
                # 나머지 _1,_2,_3 즉 직진, 우회전 차선에 해당하는 대기시간 모두 더해서 wait_time_sum_per_lane_list에 넣기.
                # wait_time_sum_per_lane_list에는 즉, 각 진입차도마다, W 직진우회전/좌회전 N 직진우회전/좌회전 E 직진우회전/좌회전 S 직진우회전/좌회전 대기시간 값 저장하게됨.
                sum_123 = df3[[direction + '_1', direction + '_2', direction + '_3']].sum().sum()
                wait_time_sum_per_lane_list.append(sum_123)

            # 이전에 리워드 변수로 선언한 차선별 대기시간을 나타내는 wait_times_per_lane 변수에 wait_time_sum_per_lane_list값 저장함.
            self.wait_times_per_lane = wait_time_sum_per_lane_list
            # current_total_wait에는 wait_time_sum_per_lane_list에 저장된 각 레인별 대기시간 모두 더해서 그냥 교차로 한개를 보았을때 모든 진입차도에 대한 대기시간을 더하게됨
            current_total_wait = sum(wait_time_sum_per_lane_list)
            self._reward_wait_time = current_total_wait
            # self._reward_wait_time = current_total_wait - previous_cycle_wait_time # 이전 사이클과 현재 사이클의 대기 시간 차이 값을 리워드 값으로 저장함. 맨처음 대기시간은 0으로 초기화 했음
            # print(f"previous_cycle_wait_time: {previous_cycle_wait_time}, current_total_wait: {current_total_wait}, _reward_wait_time: {self._reward_wait_time}")

            ########## Queue length calculate ###############
            # 대기열 계산
            df1 = df1.transpose()
            df2 = df2.transpose()
            queue_length_sum_per_lane_list = []

            # 각 방향에 대해 집계
            for direction in ['W_in', 'N_in', 'E_in', 'S_in']:
                # _0 lane - 차량 존재(df1)하고, 속도(df2)가 0.1보다 작은 경우 count
                halted_count = ((df1[direction + '_0'] == 1) & (df2[direction + '_0'] <= 0.1)).sum()
                queue_length_sum_per_lane_list.append(halted_count)

                # else lanes - 차량 존재하고, 속도가 0.1보다 작은 경우 count
                # 각 진입차도마다, W 직진우회전/좌회전 N 직진우회전/좌회전 E 직진우회전/좌회전 S 직진우회전/좌회전 대기열 값 queue_length_sum_per_lane_list에 저장하게됨.
                # 각 차선에 대해 조건 적용 후 sum
                halted_count_else = sum(
                    ((df1[direction + '_' + str(i)] == 1) & (df2[direction + '_' + str(i)] <= 0.1)).sum()
                    for i in range(1, 4)
                )
                queue_length_sum_per_lane_list.append(halted_count_else)
            
            # 이전에 리워드 변수로 선언한 차선별 대기열을 나타내는 queue_len_per_lane 변수에 queue_length_sum_per_lane_list값 저장함.
            self.queue_len_per_lane = queue_length_sum_per_lane_list
            current_total_queue_length = sum(queue_length_sum_per_lane_list)
            self._reward_queue_length = current_total_queue_length - previous_cycle_queue_length # 이전 사이클과 현재 사이클의 큐 길이 차이
            self.plot_queue_length += current_total_queue_length

            ############## To plot waiting time in one episode (total sum of waiting time in whole one episode ###################
            for lane_group in range(16):
                for lane_cell in range(100):
                    self.plot_wait_time += current_state[2][lane_group][lane_cell]


            ### reward & memory push ###
            reward = self._reward()
            # print("cycle 끝난 후 reward: ", self._reward)
            self.plot_reward += reward


            if  old_state != -1 and old_action_number != -1: #맨처음에 초기화한 값인 상태일때는 즉 첫 simulate돌기 전에는 memory에 그 값이 추가가 되지않음.
                self._ReplayMemory.push(old_state, old_action_number, current_state, reward)


            ############ Action select 및 simulate 부분 여기서 한 action이 총 40 step이 지나게 된다.(cycle) ##############
            action_to_do = self._choose_action(current_state, epsilon)
            print(f"Selected Action: {action_to_do} with ratios {self._action_ratios[action_to_do]}")  # 액션 출력
            ratios = self._action_ratios[action_to_do]
            for i, duration in enumerate(ratios):
                green_phase_code = self.phases[i * 2]
                yellow_phase_code = self.phases[i * 2 + 1]
                traci.trafficlight.setPhase("intersection", green_phase_code)
                self._simulate(duration)
                traci.trafficlight.setPhase("intersection", yellow_phase_code)
                self._simulate(3)
            # 40 step이 지나고 1 cycle 종료
            #######################################################################################################

            
            self.optimize_model()

            old_state = current_state
            old_action_number = action_to_do
            previous_cycle_queue_length = current_total_queue_length  # 이전 사이클의 큐 길이 업데이트
            # previous_cycle_wait_time = current_total_wait  # 이전 사이클의 대기 시간 업데이트

        if len(self.veh_wait_time_in_lane)==1000:
            with open('dictionary_values.txt', 'w') as file:
                for key, value in self.veh_wait_time_in_lane.items():
                    file.write(f"{key}: {value}\n")
            veh_total_wait_sum=sum(self.veh_wait_time_in_lane.values())
            num_cars=len(self.veh_wait_time_in_lane)
            average_wait_time=veh_total_wait_sum/num_cars if num_cars>0 else 0

        self.reward_per_episode.append(self.plot_reward / 90)
        self._waiting_time_per_episode.append(average_wait_time)
        self._queue_length_per_episode.append(self.plot_queue_length / 90)
        print(f"epsilon : {epsilon:.3f}")

        traci.close()
        

    # def _get_state(self):
    #     state = np.zeros((3, 16, 100))
    #     car_list = traci.vehicle.getIDList()

    #     for car_id in car_list:
    #         lane_pos = 750 - traci.vehicle.getLanePosition(car_id)
    #         lane_id = traci.vehicle.getLaneID(car_id)
    #         lane_pos = min(999, lane_pos)
    #         lane_group = -1

    #         if lane_id in lane:
    #             lane_cell = min(99, int(lane_pos / 7.5))
    #             for idx in range(len(lane)):
    #                 if lane_id == lane[idx]:
    #                     lane_group = idx

    #             state[0][lane_group][lane_cell] = 1
    #             state[1][lane_group][lane_cell] = traci.vehicle.getSpeed(car_id)
    #             state[2][lane_group][lane_cell] = traci.vehicle.getAccumulatedWaitingTime(car_id)

    #     return state.tolist()
    def _get_state(self):
        state=np.zeros((3,16,100))
        vehicle_list=traci.vehicle.getIDList()
        lane_group=0
        
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
    
    def _choose_action(self, state, epsilon):
        state_tensor = torch.tensor([state], device=device, dtype=torch.float)
        if random.random() < epsilon:
            action = random.choice(range(self._num_actions))
        else:
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                action = q_values.max(1)[1].item()
        return action

    def _simulate(self, steps_todo):
        if (self._step + steps_todo) >= self._max_steps:
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()
            self._step += 1
            steps_todo -= 1
            ###vehicle time calculate###
            vehicle_list=traci.vehicle.getIDList()
            for veh_id in vehicle_list:
                lane_id = traci.vehicle.getLaneID(veh_id)  # 차선 ID
                # print(f"Vehicle ID: {veh_id}, Lane ID: {lane_id}")  # 디버깅을 위한 출력
        
                if lane_id.startswith('E_in_') or lane_id.startswith('N_in_') or lane_id.startswith('S_in_') or lane_id.startswith('W_in_'):
                    if veh_id not in self.veh_time_in_lane:
                        self.veh_time_in_lane[veh_id] = []

                    self.veh_time_in_lane[veh_id].append(self._step)
                else:
                    if veh_id in self.veh_time_in_lane:
                        self.veh_wait_time_in_lane[veh_id]=self.veh_time_in_lane[veh_id][-1]-self.veh_time_in_lane[veh_id][0]
                if self._step==self._max_steps: #episode가 끝날때까지 진입차선을 벗어나지 못한 차들을 그냥 3600-진입차선에 들어간 시점 으로 값 할당하기
                    self.veh_wait_time_in_lane[veh_id]=self.veh_time_in_lane[veh_id][-1]-self.veh_time_in_lane[veh_id][0]

    def optimize_model(self):

        if len(self._ReplayMemory) < BATCH_SIZE:
            return

        experience = self._ReplayMemory.sample(BATCH_SIZE)
        batch = Experience(*zip(*experience))

        # 최종이 아닌 상태의 마스크를 계산하고 배치 요소를 연결합니다
        # (최종 상태는 시뮬레이션이 종료 된 이후의 상태)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        # non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).view(-1,3,48,100).to(device)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).view(-1, 3, 16, 100).to(
            device)

        # non_final_next_states = torch.cat([torch.tensor(s, dtype=torch.float).unsqueeze(0) for s in batch.next_state if s is not None]).to(device)

        # state_batch = torch.cat(batch.state).view(BATCH_SIZE,3,48,100).to(device)
        state_batch = torch.cat(batch.state).view(BATCH_SIZE, 3, 16, 100).to(device)
        action_batch = torch.cat(batch.action).view(-1, 1).to(device)
        reward_batch = torch.cat(batch.reward).to(device)
        # prev_actions_batch = torch.cat(batch.prev_actions).view(BATCH_SIZE, -1).to(device)  # prev_actions 처리

        # print('prev_actions_batch',prev_actions_batch)

        q_eval = self.policy_net(state_batch).gather(1, action_batch)
        # q_eval = self.policy_net(state_batch, prev_actions_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=device)

        with torch.no_grad():
            # next_state_values[non_final_mask] = self.target_net(non_final_next_states,prev_actions_batch[non_final_mask]).max(1)[0].detach()
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

        # update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.learn_step_counter += 1

    def _reward(self):
        w_1 = 1
        w_2 = 0
        w_3 = 0

        waiting_time = self._reward_wait_time
        queue_length = self._reward_queue_length

        avg_waiting_time = 1
        avg_queue_length = 1.5

        each_waiting_time_for_fairness = self.wait_times_per_lane
        each_queue_length_for_fairness = self.queue_len_per_lane
        waiting_time_fairness = self.calculate_fairness_index(each_waiting_time_for_fairness)
        queue_length_fairness = self.calculate_fairness_index(each_queue_length_for_fairness)
        reward= -(w_1*waiting_time/avg_waiting_time + w_2*queue_length/avg_queue_length) + w_3*(w_1/(w_1+w_2)*waiting_time_fairness+ w_2/(w_1+w_2)*queue_length_fairness)

        print(f"Reward calculation - waiting_time: {waiting_time}, queue_length: {queue_length}, reward: {reward}")

        return reward

    def calculate_fairness_index(self, values):
        if not values:
            return 1.0
        square_of_sums = np.square(np.sum(values))
        sum_of_squares = np.sum(np.square(values))
        if sum_of_squares > 0:
            return square_of_sums / (len(values) * sum_of_squares)
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
        return self.reward_per_episode

