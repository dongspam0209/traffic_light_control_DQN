from collections import deque
from collections import namedtuple 
import torch
import random
Experience = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity=capacity
        self.memory: deque[Experience] = deque([], maxlen=capacity)

    def push(self, state, action, next_state, reward):
        """state, action, next_state, reward를 tensor로 변환하여 저장"""
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0) if not torch.is_tensor(state) else state
        action = torch.tensor([action], dtype=torch.long)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0) if not torch.is_tensor(next_state) else next_state
        reward = torch.tensor([reward], dtype=torch.float32)
        
        # 변환된 tensor를 Experience 객체에 저장
        self.memory.append(Experience(state, action, next_state, reward))

    def sample(self, batch_size:int)->list[Experience]:
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)