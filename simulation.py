import traci
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from replay import Experience
import itertools

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
        self._num_actions = len(self._action_ratios)

        self.policy_net = DQN(num_states, self._num_actions).to(device)
        self.target_net = DQN(num_states, self._num_actions).to(device)

        self.learn_step_counter = 0
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)

        self._ReplayMemory = ReplayMemory
        self._Cargenerator = Cargenerator
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._num_states = num_states

        # Traffic signal duration
        self._green_duration = green_duration
        self._green_turn_duration = green_turn_duration

        self.phases = [PHASE_NS_GREEN, PHASE_NS_YELLOW, PHASE_NSL_GREEN, PHASE_NSL_YELLOW,
                       PHASE_EW_GREEN, PHASE_EW_YELLOW, PHASE_EWL_GREEN, PHASE_EWL_YELLOW]
        self.current_phase_index = 0

        # Plot variables
        self._queue_length_per_episode = []
        self._waiting_time_per_episode = []
        self.reward_per_episode = []
        self.loss_history = []
        self.max_q_value_per_episode = []

    def run(self, episode, epsilon):
        if self._Cargenerator is not None:
            self._Cargenerator.generate_car(seed=episode)  # Car generation

        traci.start(self._sumo_cmd)
        print("Simulating")

        # Plot variables
        self.plot_queue_length = 0
        self.plot_wait_time = 0
        self.plot_reward = 0
        cycle_count = 0

        # Reward variables
        self.queue_len_per_lane = []  # Lane별 queue 길이
        self._reward_queue_length = 0  # Queue 길이 총합

        self.wait_times_per_lane = []
        self._reward_wait_time = 0

        # Inits
        self._step = 0

        previous_cycle_queue_length = 0  # 이전 사이클의 큐 길이
        previous_cycle_wait_time = 0  # 이전 사이클의 대기 시간

        while self._step < self._max_steps:
            ############# Get state ##################
            current_state = self._get_state()
            df1 = pd.DataFrame(current_state[0], index=lane)
            df1.to_csv('./intersection/generate_exist.csv')
            df2 = pd.DataFrame(current_state[1], index=lane)
            df2.to_csv('./intersection/generate_velocity.csv')
            df3 = pd.DataFrame(current_state[2], index=lane)
            df3.to_csv('./intersection/generate_waiting_time.csv')

            ########## Waiting time calculate ###############
            df3 = df3.transpose()
            wait_time_sum_per_lane_list = []

            # 각 방향에 대해 집계
            for direction in ['W_in', 'N_in', 'E_in', 'S_in']:
                # W_in_0
                wait_time_sum_per_lane_list.append(df3[direction + '_0'].sum())

                # else lanes
                sum_123 = df3[[direction + '_1', direction + '_2', direction + '_3']].sum().sum()
                wait_time_sum_per_lane_list.append(sum_123)

            self.wait_times_per_lane = wait_time_sum_per_lane_list
            current_total_wait = sum(wait_time_sum_per_lane_list)
            self._reward_wait_time = previous_cycle_wait_time - current_total_wait  # 이전 사이클과 현재 사이클의 대기 시간 차이

            ########## Queue length calculate ###############
            df1 = df1.transpose()
            df2 = df2.transpose()
            queue_length_sum_per_lane_list = []

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

            self.queue_len_per_lane = queue_length_sum_per_lane_list
            current_total_queue_length = sum(queue_length_sum_per_lane_list)
            self._reward_queue_length = previous_cycle_queue_length - current_total_queue_length  # 이전 사이클과 현재 사이클의 큐 길이 차이
            self.plot_queue_length += current_total_queue_length

            ############## To plot waiting time in one episode (total sum of waiting time in whole one episode ###################
            for lane_group in range(16):
                for lane_cell in range(100):
                    self.plot_wait_time += current_state[2][lane_group][lane_cell]

            ############ Action select ##############
            action_to_do = self._choose_action(current_state, epsilon)
            print(f"Selected Action: {action_to_do} with ratios {self._action_ratios[action_to_do]}")  # 액션 출력

            old_state = current_state
            old_action_number = action_to_do

            ratios = self._action_ratios[action_to_do]

            for i, duration in enumerate(ratios):
                green_phase_code = self.phases[i * 2]
                yellow_phase_code = self.phases[i * 2 + 1]

                traci.trafficlight.setPhase("intersection", green_phase_code)
                self._simulate(duration)

                traci.trafficlight.setPhase("intersection", yellow_phase_code)
                self._simulate(3)

            cycle_count += 1

            new_state = self._get_state()
            reward = self._reward()
            self.plot_reward += reward

            if old_state is not None:
                self._ReplayMemory.push(old_state, old_action_number, new_state, reward)
                self.optimize_model()

            previous_cycle_queue_length = current_total_queue_length  # 이전 사이클의 큐 길이 업데이트
            previous_cycle_wait_time = current_total_wait  # 이전 사이클의 대기 시간 업데이트

        if cycle_count > 0:
            self.reward_per_episode.append(self.plot_reward / cycle_count)
            self._waiting_time_per_episode.append(self.plot_wait_time / self._step)
            self._queue_length_per_episode.append(self.plot_queue_length / self._step)
        else:
            self.reward_per_episode.append(0)
            self._waiting_time_per_episode.append(0)
            self._queue_length_per_episode.append(0)

        print(f"epsilon : {epsilon:.3f}")

        traci.close()
        return

    def _get_state(self):
        state = np.zeros((3, 16, 100))
        car_list = traci.vehicle.getIDList()

        for car_id in car_list:
            lane_pos = 750 - traci.vehicle.getLanePosition(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)
            lane_pos = min(999, lane_pos)
            lane_group = -1

            if lane_id in lane:
                lane_cell = min(99, int(lane_pos / 7.5))
                for idx in range(len(lane)):
                    if lane_id == lane[idx]:
                        lane_group = idx

                state[0][lane_group][lane_cell] = 1
                state[1][lane_group][lane_cell] = traci.vehicle.getSpeed(car_id)
                state[2][lane_group][lane_cell] = traci.vehicle.getAccumulatedWaitingTime(car_id)

        return state.tolist()

    def _choose_action(self, state, epsilon):
        state_tensor = torch.tensor([state], device=device, dtype=torch.float)
        if random.random() < epsilon:
            action = random.choice(range(self._num_actions))
        else:
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                action = q_values.max(1)[1].item()
                self.max_q_value_per_episode.append(q_values.max().item())
        return action

    def _simulate(self, steps_todo):
        if (self._step + steps_todo) >= self._max_steps:
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()
            self._step += 1
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

        avg_waiting_time = 100
        avg_queue_length = 1.5

        each_waiting_time_for_fairness = self.wait_times_per_lane
        each_queue_length_for_fairness = self.queue_len_per_lane
        waiting_time_fairness = self.calculate_fairness_index(each_waiting_time_for_fairness)
        queue_length_fairness = self.calculate_fairness_index(each_queue_length_for_fairness)

        reward = -(w_1 * waiting_time)
        # reward = -(w_1 * waiting_time / avg_waiting_time + w_2 * queue_length / avg_queue_length) + w_3 * (
        #      w_1 / (w_1 + w_2) * waiting_time_fairness + w_2 / (w_1 + w_2) * queue_length_fairness)

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

