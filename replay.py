from collections import deque
from collections import namedtuple 
import torch
import numpy as np
from random import sample

device = "cuda" if torch.cuda.is_available() else "cpu"
Experience = namedtuple("Experience", ("state", "action", "next_state", "reward"))


class ReplayMemory:

    def __init__(self, capacity : int) -> None:
        self.capacity = capacity
        self.memory: deque[Experience] = deque([], maxlen = capacity)

    def __len__(self) -> int:

        return self.memory.__len__() # 저장된 경험 수 리턴

    def push(self, experience: Experience) -> None:
 
        self.memory.append(experience) # 새로운 경험을 저장. 오래된 경험부터 덮어씀

    def sample(self, batch_size: int) -> list[Experience]:

        return np.random.sample(self.memory, batch_size) # 몇몇 경험을 랜덤하게 샘플링함