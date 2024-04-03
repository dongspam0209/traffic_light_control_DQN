from collections import deque
from collections import namedtuple 
import torch
import random
Experience = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity=capacity
        self.memory = deque([], maxlen=capacity)
        self.temp_memory = deque([], maxlen=3)
        self.GAMMA = 0.90

    def push(self, state, action, next_state, reward):
        """state, action, next_state, reward를 tensor로 변환하여 저장"""
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0) if not torch.is_tensor(state) else state
        action = torch.tensor([action], dtype=torch.long) if not torch.is_tensor(action) else action
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0) if not torch.is_tensor(next_state) else next_state
        reward = torch.tensor([reward], dtype=torch.float32) if not torch.is_tensor(reward) else reward

        self.temp_memory.append(Experience(state, action, next_state, reward))

        if len(self.temp_memory) < 3:
            return

        state, action, _, _ = self.temp_memory[0]
        _, _, _, reward = self.temp_memory[-1]
        rewards = sum(exp.reward * (self.GAMMA ** i) for i, exp in enumerate(self.temp_memory))
        print("Last step reward: ", reward.item(), "/ 3-step total rewards: ", rewards.item())

        self.memory.append(Experience(state, action, next_state, rewards))

        self.temp_memory.popleft()
    def sample(self, batch_size:int)->list[Experience]:
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)