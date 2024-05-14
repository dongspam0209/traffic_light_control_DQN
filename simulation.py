import traci
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from replay import Experience
import itertools

###################################################################
# hyper-parameters
BATCH_SIZE = 128
GAMMA = 0.90
LR = 1e-4
MEMORY_CAPACITY = 10000
Q_NETWORK_ITERATION = 10

# action phase definition
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

###################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Simulation:
    def __init__(self, DQN, ReplayMemory, Cargenerator, sumo_cmd, max_steps, num_states, _num_actions, green_duration,
                 green_turn_duration, cycle_duration=20):

        self._cycle_duration = cycle_duration  # 사이클 총 지속 시간, 예: 20 스텝
        self._action_ratios = list(itertools.product([1, 2], repeat=4))  # Generates 16 combinations
        self._num_actions = len(set(self._action_ratios))  # Total number of action combinations

        self.policy_net = DQN(num_states, self._num_actions).to(device)
        self.target_net = DQN(num_states, self._num_actions).to(device)

        self.learn_step_counter = 0
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)

        self._ReplayMemory = ReplayMemory
        self._Cargenerator = Cargenerator
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._num_states = num_states

        # traffic signal duration
        self._green_duration = green_duration
        self._green_turn_duration = green_turn_duration

        self.phases = [PHASE_NS_GREEN, PHASE_NS_YELLOW, PHASE_NSL_GREEN, PHASE_NSL_YELLOW,
                       PHASE_EW_GREEN, PHASE_EW_YELLOW, PHASE_EWL_GREEN, PHASE_EWL_YELLOW]
        self.current_phase_index = 0

        # plot variables
        self._queue_length_per_episode = []
        self._waiting_time_per_episode = []
        self.reward_per_epsiode = []
        self.loss_history = []
        self.max_q_value_per_episode = []

    def run(self, episode, epsilon):
        self._Cargenerator.generate_car(seed=episode)  # car generation
        traci.start(self._sumo_cmd)
        print("Simulating")

        # plot variables
        self.plot_queue_length = 0
        self.plot_wait_time = 0
        self.plot_reward = 0

        # reward variables
        self.queue_len_per_lane = []  # lane별 queue 길이
        self._reward_queue_length = 0  # queue 길이 총합

        self.wait_times_per_lane = []
        self._reward_wait_time = 0

        # inits
        self._step = 0
        old_action_number = -1
        old_total_wait = 0
        old_state = 0
        # # previous state #
        # self.old_state_history = [0, 0, 0]
        # self.state_history= np.zeros((3,3,16,100))
        # self.state_history=self.state_history.tolist()
        # # previous action #
        # self.action_history = [-1, -1]

        while self._step < self._max_steps:
            ############# get state ##################
            current_state = self._get_state()
            df1 = pd.DataFrame(current_state[0], index=lane)
            df1.to_csv('./intersection/generate_exist.csv')
            df2 = pd.DataFrame(current_state[1], index=lane)
            df2.to_csv('./intersection/generate_velocity.csv')
            df3 = pd.DataFrame(current_state[2], index=lane)
            df3.to_csv('./intersection/generate_waiting_time.csv')

            # if len(self.state_history) >= 3:
            #     self.state_history.pop(0)
            # self.state_history.append(current_state)
            # # state_history (t+2,t+1,t) 제일 끝이 제일 최근 것임. 따라서, 제일 이전에 관측한 state가 먼저 탈출함. pop(0)로 첫번째 데이터(제일 과거의 state)제거함.

            ########## waiting time calculate ###############
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
            self._reward_wait_time = current_total_wait - old_total_wait

            #################################################

            ########## queue_length calculate ###############
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
            self._reward_queue_length = sum(queue_length_sum_per_lane_list)
            self.plot_queue_length += self._reward_queue_length
            #################################################

            ############## to plot waiting time in one episode (total sum of waiting time in whole one episode ###################
            for lane_group in range(16):
                for lane_cell in range(100):
                    self.plot_wait_time += current_state[2][lane_group][lane_cell]

            ############ action select ##############
            # tensor_list = [torch.tensor(x, device=device, dtype=torch.float) for x in self.state_history]
            # state_cat = torch.cat(tensor_list, dim=1)

            # prev_actions_tensor = torch.tensor(self.action_history, dtype=torch.long).unsqueeze(0).to(device)

            # action_to_do=self._choose_action(state_cat,prev_actions_tensor,epsilon) # action_to_do : a_t
            action_to_do = self._choose_action(current_state, epsilon)
            print(f"Selected Action: {action_to_do} with ratios {self._action_ratios[action_to_do]}")  # 액션 출력
            self._set_green_phase(action_to_do)
            # if len(self.action_history) >= 2: # action index 저장
            #     self.action_history.pop(0)
            # self.action_history.append(action_to_do)

            old_state = current_state
            old_action_number = action_to_do
            old_total_wait = current_total_wait

            ############# reward & memory push #######
            reward = self._reward()
            self.plot_reward += reward

            if self._step != 0:
                # self._ReplayMemory.push(self.old_state_history, old_action_number,state_cat,reward,self.action_history)
                self._ReplayMemory.push(old_state, old_action_number, current_state, reward)
            ###########################################


        self.reward_per_epsiode.append(self.plot_reward / self._step)
        self._waiting_time_per_episode.append(self.plot_wait_time / self._step)
        self._queue_length_per_episode.append(self.plot_queue_length / self._step)

        print(f"epsilon : {epsilon:.3f}")

        traci.close()

    def _get_state(self):
        state = np.zeros((3, 16, 100))
        vehicle_list = traci.vehicle.getIDList()
        lane_group = 0

        for veh_id in vehicle_list:
            lane_position = traci.vehicle.getLanePosition(veh_id)  # car position in lane
            lane_id = traci.vehicle.getLaneID(veh_id)  # -> lane
            # print(lane_id)
            lane_position = 700 - lane_position  # traffic light ~> max len of road 0~700
            lane_cell = int(lane_position / 7)  # 7m : hegiht of the cell
            lane_cell = min(lane_cell, 99)

            for idx in range(len(lane)):
                if lane_id == lane[idx]:
                    lane_group = idx

            state[0][lane_group][lane_cell] = 1  # cell occupied
            state[1][lane_group][lane_cell] = traci.vehicle.getSpeed(veh_id)  # vehicle velocity
            state[2][lane_group][lane_cell] = traci.vehicle.getAccumulatedWaitingTime(veh_id)  # waiting time

        return state.tolist()

    def _set_green_phase(self, action_number):
        """Set traffic light phases based on selected action, handle both green and yellow phases."""
        ratios = self._action_ratios[action_number]
        total_ratio = sum(ratios)
        print(f"total_ratio: {total_ratio}")
        total_available_duration = 20 - len(ratios)  # Assuming 1s for each yellow phase
        # Using rounding instead of truncating to integer
        raw_durations = [round(total_available_duration * ratio / total_ratio) for ratio in ratios]
        print(f"raw_durations: {raw_durations}")
        # Each green phase is followed immediately by a yellow phase
        for i, duration in enumerate(raw_durations):
            green_phase_code = self.phases[i * 2]
            yellow_phase_code = self.phases[i * 2 + 1]

            # Set green phase
            traci.trafficlight.setPhase("intersection", green_phase_code)
            print(f"Setting green phase {green_phase_code} for {duration} steps")
            self._simulate(duration)

            # Set yellow phase, assuming fixed duration of 1 step for simplicity
            traci.trafficlight.setPhase("intersection", yellow_phase_code)
            print(f"Setting yellow phase {yellow_phase_code} for 1 step")
            self._simulate(1)

        # After setting all phases, a complete cycle has ended, so optimize model
        self._optimize_model()
        return sum(raw_durations) + len(raw_durations)  # Include yellow phases in total duration

    # def _choose_action(self, state, prev_actions_tensor, epsilon):
    #     """
    #     Decide wheter to perform an explorative or exploitative action, according to an epsilon-greedy policy
    #     """
    #     state_tensor = state
    #     if random.random() < epsilon:
    #         # print("explore")
    #         return random.randint(0, self._num_actions - 1) # random action
    #     else:
    #         with torch.no_grad():
    #             # print("exploit")
    #             out = self.policy_net(state_tensor.unsqueeze(0),prev_actions_tensor)
    #             max_val_list, max_q_action_list = torch.max(out, dim=1)

    #             max_index = torch.argmax(max_val_list) # 가장 큰 값의 index 추출
    #             real_action = max_q_action_list[max_index] # 그 index에 해당하는 action 번호 추출
    #             # print("max_val_list :",max_val_list,"max_q_action_list:",max_q_action_list)
    #             # print("max Q value selected",real_action)

    #             return real_action.item()

    def _choose_action(self, state, epsilon):
        state_tensor = torch.tensor([state], device=device, dtype=torch.float)
        if random.random() < epsilon:
            action = random.choice(range(self._num_actions))
        else:
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                action = q_values.max(1)[1].item()
                self.max_q_value_per_episode.append(q_values.max().item())  # Track max Q value for this step
        return action

    def _simulate(self, steps_todo):
        """
        Execute steps in sumo while gathering statistics
        """
        if (self._step + steps_todo) >= self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._step += 1  # update the step counter
            steps_todo -= 1

    def _optimize_model(self):
        # Check if we have enough samples in the memory to get a full batch
        if len(self._ReplayMemory) < BATCH_SIZE:
            return

        # Sample a batch from the memory
        transitions = self._ReplayMemory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for detailed explanation).
        batch = Experience(*zip(*transitions))

        # Convert batch-array of Experiences to Experience of batch-arrays.
        state_batch = torch.cat(batch.state).view(BATCH_SIZE, 3, 16, 100).to(device)
        action_batch = torch.cat(batch.action).view(-1, 1).to(device)
        reward_batch = torch.cat(batch.reward).to(device)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).view(-1,3,16,100).to(device)

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device,
                                      dtype=torch.bool)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        q_eval = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        # Compute the expected Q values
        q_target = (next_state_values * GAMMA) + reward_batch

        criterion = nn.MSELoss()
        loss = criterion(q_eval, q_target.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss_history.append(loss.item())

        # update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.learn_step_counter += 1

    def _reward(self):
        w_1 = 1 / 3
        w_2 = 1 / 3
        w_3 = 1 / 3

        waiting_time = self._reward_wait_time
        queue_length = self._reward_queue_length

        avg_waiting_time = 100
        avg_queue_length = 1.5

        each_waiting_time_for_fairness = self.wait_times_per_lane
        each_queue_length_for_fairness = self.queue_len_per_lane
        waiting_time_fairness = self.calculate_fairness_index(each_waiting_time_for_fairness)
        queue_length_fairness = self.calculate_fairness_index(each_queue_length_for_fairness)

        reward = -(w_1 * waiting_time / avg_waiting_time + w_2 * queue_length / avg_queue_length) + w_3 * (
                    w_1 / (w_1 + w_2) * waiting_time_fairness + w_2 / (w_1 + w_2) * queue_length_fairness)

        return reward

    # fairness index 0~1
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
        return self.reward_per_epsiode

    @property
    def max_q_value(self):
        return self.max_q_value_per_episode
