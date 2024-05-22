from collections import deque,namedtuple
import torch
import random
Experience = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
# Experience = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'prev_actions'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity=capacity
        self.memory: deque[Experience] = deque([], maxlen=capacity)

    # def push(self, state, action, next_state, reward,prev_actions):
    def push(self, state, action, next_state, reward):
        """state, action, next_state, reward를 tensor로 변환하여 저장"""
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0) if not torch.is_tensor(state) else state
        action = torch.tensor([action], dtype=torch.long)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0) if not torch.is_tensor(next_state) else next_state
        reward = torch.tensor([reward], dtype=torch.float32)
        # prev_actions_tensor = torch.tensor(prev_actions, dtype=torch.long).unsqueeze(0)

        # self.memory.append(Experience(state, action, next_state, reward, prev_actions_tensor))
        self.memory.append(Experience(state, action, next_state, reward))


    def sample(self, batch_size:int)->list[Experience]:
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
from collections import deque,namedtuple
import torch
import random
Experience = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
# Experience = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'prev_actions'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity=capacity
        self.memory: deque[Experience] = deque([], maxlen=capacity)

    # def push(self, state, action, next_state, reward,prev_actions):
    def push(self, state, action, next_state, reward):
        """state, action, next_state, reward를 tensor로 변환하여 저장"""
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0) if not torch.is_tensor(state) else state
        action = torch.tensor([action], dtype=torch.long)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0) if not torch.is_tensor(next_state) else next_state
        reward = torch.tensor([reward], dtype=torch.float32)
        # prev_actions_tensor = torch.tensor(prev_actions, dtype=torch.long).unsqueeze(0)

        # self.memory.append(Experience(state, action, next_state, reward, prev_actions_tensor))
        self.memory.append(Experience(state, action, next_state, reward))


    def sample(self, batch_size:int)->list[Experience]:
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)