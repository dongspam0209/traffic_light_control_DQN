import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, n_states, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(n_states[0], 16, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=1)

        # self.fc1 = nn.Linear(9600, 512)
        
        self.fc1 = nn.Linear(32*16*100, 512)

        self.fc2 = nn.Linear(512, n_actions)

    # def forward(self, x, prev_action):
    def forward(self, x):

        # print("original size: ", x.size())
        x = F.relu(self.conv1(x))
        # print("conv1 후 size: ", x.size())
        x = F.relu(self.conv2(x))
        # print("conv2 후 size: ", x.size())

        x = x.view(x.size(0), -1)
        # print("2차원으로 변환: ", x.size())
        # print('model에서의 prev_action',prev_action) #prev_action은 128개 case에 대해서, 각각 이전 action 두개를 포함해야함.
        # x=torch.cat((x,prev_action),dim=1)
        # print('이전 action 추가한 shape; ',x.size())

        x = F.relu(self.fc1(x))
        # print("fc_layer 1번 거침: ", x.size())
        x = self.fc2(x)
        # print("최종 output size: ", x.size())
        return x