from collections import deque
from collections import namedtuple 
import torch
import random
import numpy as np
from random import sample

# Experience = namedtuple("Experience", ("state", "action", "next_state", "reward"))


# class ReplayMemory:

#     def __init__(self, capacity : int) -> None:
#         self.capacity = capacity
#         self.memory: deque[Experience] = deque([], maxlen = capacity)

#     def __len__(self) -> int:

#         return self.memory.__len__() # 저장된 경험 수 리턴

#     def push(self, experience: Experience) -> None:
 
#         self.memory.append(experience) # 새로운 경험을 저장. 오래된 경험부터 덮어씀

#     def sample(self, batch_size: int) -> list[Experience]:

#         return np.random.sample(self.memory, batch_size) # 몇몇 경험을 랜덤하게 샘플링함
Experience = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity=capacity
        self.memory: deque[Experience] = deque([], maxlen=capacity)

    # def push(self, experience: Experience)->None:
    #     """transition 저장"""
    #     self.memory.append(experience)
    def push(self, state, action, next_state, reward):
        """state, action, next_state, reward를 tensor로 변환하여 저장"""
        # 여기서는 각 값을 tensor로 변환하고 device에 맞게 설정합니다.
        # 예를 들어, device를 'cuda'로 설정했다면 .to('cuda')를 추가합니다.
        # state, next_state의 차원과 데이터 타입은 실제 환경에 맞게 조정해야 합니다.
        state = torch.tensor([state], dtype=torch.float32)
        action = torch.tensor([action], dtype=torch.long)
        next_state = torch.tensor([next_state], dtype=torch.float32)
        reward = torch.tensor([reward], dtype=torch.float32)
        
        # 변환된 tensor를 Experience 객체에 저장
        self.memory.append(Experience(state, action, next_state, reward))

    def sample(self, batch_size:int)->list[Experience]:
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)