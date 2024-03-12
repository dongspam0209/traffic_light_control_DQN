import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"


def normalize_output(outputs): # 출력 정규화 
    with torch.no_grad():
        for m, output in enumerate(outputs):
            max_t = torch.max(torch.abs(output))
            if abs(max_t-0) < 1e-2:
                continue
            for n, t in enumerate(output):
                outputs[m][n] /= max_t
    return outputs


class DQN(nn.Module):
    def __init__(self, inputs, outputs, hidden_dims = [512, 256], activation = F.relu):
        super(DQN, self).__init__()
        self.input_dim = inputs
        self.output_dim = outputs
        self.hidden_dim = hidden_dims
        self.activation = activation

        layers = []
        dims = [self.input_dim] + self.hidden_dims # 입력층 + 은닉층 합친 리스트

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(self.activation())

        # 입력층 ~ 은닉층까지 선형 변환 층 및 활성화 함수 리스트에 추가
            
        layers.append(nn.Linear(dims[-1], self.output_dim)) # 마지막에 출력층 추가

        self.model = nn.Sequential(*layers) # 추가한 층들을 연결하여 모델 생성

        self.init_weights() 

    def init_weights(self): # 가중치 초기화 함수
        for layer in self.model:
            if isinstance(layer, nn.Linear): # 각 층이 선형 변환층일 시
                c = np.sqrt(1 / layer.in_features) # 초기화를 위한 변수 선언
                nn.init.uniform_(layer.weight, -c, c) 
                nn.init.uniform_(layer.bias, -c, c) # 변수로 weight, bias 균등 분포로 초기호ㅏ

    def forward(self, x): # 순전파 함수임
        x = x.to(device) # gpu 관련
        x = self.model(x) # 모델에 입력 전달
        return x # 출력을 반환시킴

# 각 변수 설정하는 파트
input_dim = 10
output_dim = 4
hidden_dims = [512, 256]
activation = F.relu  

# 모델 생성하는 파트
improved_model = DQN(input_dim, output_dim, hidden_dims=hidden_dims, activation=activation)



# class DQN(nn.Module):
#     def __init__(self, input_d, hidden_d, hidden_act, output_d, output_act, lr, lre):
#         super(DQN, self).__init__()

#         layers = []
#         layers.append(nn.Linear(input_d, hidden_d[0]))
#         layers.append(nn.ReLU() if hidden_act == 'relu' else nn.Identity())
#         for i in range(1, len(hidden_d)):
#             layers.append(nn.Linear(hidden_d[i - 1], hidden_d[i]))
#             layers.append(nn.ReLU() if hidden_act == 'relu' else nn.Identity())

#         layers.append(nn.Linear(hidden_d[-1], output_d))
#         layers.append(nn.Identity() if output_act == 'linear' else nn.ReLU())

#         self.models = {
#             'online': nn.Sequential(*layers),
#             'target': nn.Sequential(*layers)
#         }

#     def forward(self, _input, nettype):
#         return self.models[nettype](_input)

#     def backward(self, _input, _target):
#         optimizer = optim.Adam(self.models['online'].parameters())
#         criterion = nn.MSELoss()

#         input_tensor = torch.tensor(_input, dtype=torch.float32)
#         target_tensor = torch.tensor(_target, dtype=torch.float32)

#         optimizer.zero_grad()
#         output = self.models['online'](input_tensor)
#         loss = criterion(output, target_tensor)
#         loss.backward()
#         optimizer.step()

#     def transfer_weights(self):
#         self.models['target'].load_state_dict(self.models['online'].state_dict())

#     def get_weights(self, nettype):
#         return self.models[nettype].state_dict()

#     def set_weights(self, weights, nettype):
#         self.models[nettype].load_state_dict(weights)

#     def save_weights(self, nettype, path, fname):
#         torch.save(self.models[nettype].state_dict(), os.path.join(path, fname + '.pth'))

#     def load_weights(self, path):
#         path += '.pth'
#         if os.path.exists(path):
#             weights = torch.load(path)
#             self.models['online'].load_state_dict(weights)
#         else:
#             assert 0, 'Failed to load weights, supplied weight file path ' + str(path) + ' does not exist.'


# if __name__ == '__main__':
#     input_d = 10
#     dqn = DQN(input_d, [20, 20], 'relu', 4, 'linear', 0.0001, 0.0000001)

#     x = torch.rand((1, input_d), dtype=torch.float32)
#     output = dqn.forward(x, 'online')
#     print(output)
