# 🚗 DQN 기반 왕복 8차선 traffic light control system

# 목차
1. [프로젝트 소개](#1-프로젝트-소개)
2. [프로젝트 진행](#2-프로젝트-진행)
3. [실험 결과](#3-실험-결과)
4. [Contact](#4-Contact)

# 1. 프로젝트 소개
(1) 주제 및 목표
- 오늘날 사회 대부분의 신호등은 신호의 순서와 주기가 정해진 고정형 신호체계를 채택하고 있다. 그러나 이는 급변하는 교통 변화의 흐름에 유동적으로 대응하지 못한다는 단점이 있기에 이러한 문제 해결을 위해 강화학습 기반 적응형 신호 체계가 연구되고 있다. 본 프로젝트에서는 이를 더욱 개선하고자 현실 조건을 반영하고 여러 기법을 비교하며 최적의 신호 체계를 고안하였다. 

(2) 평가지표
- 교차로 내 차량 대기시간
- 교차로 내 차량 대기열

(3) 개발 환경  
- GPU : GTX 1070Ti
- SUMO : 교통신호제어 library
- state : waiting time, queue length
- action : traffic light selection
- reward : reduced waiting time, queue length difference after action

# 2. 프로젝트 진행
## 1. Simulation Environment
| **1** | **2** | **3** | **4** |
|------------|------------|------------|------------|
| ![image](https://github.com/user-attachments/assets/dbf4378b-1414-45e8-acf2-e37d00622ee1) | ![image](https://github.com/user-attachments/assets/2d17819b-58b5-4bfb-bbc3-f00a4fb2fa97) | ![Step 3](https://github.com/user-attachments/assets/81fdcd66-283d-47af-9cbd-b23e2a71b3aa) | ![Step 4](https://github.com/user-attachments/assets/87a7a0d8-fc32-4768-aa48-afa0af1d7f19) |
1. 한 에피소드당 1000대의 차량 생성
2. 각 교차로의 진입차도에서 1차선은 좌회전, 2,3차선은 직진, 4차선은 직진 또는 우회전으로 구성
3. 실제 교통량이 늘어나는 시간대를 구현하기 위하여, 로그정규분포를 통해 진입차도별로 차량이 많이 생성되는 시간대 설정
4. 차량의 크기는 5m(default) 앞뒤 차 간격은 2.5m

## 2. 차선 구역화
![image](https://github.com/user-attachments/assets/7316691e-da5f-4af4-95ca-d6c4ea0e931a)
- 각 진입차선을 100개의 cell로 구역화하여 1초간격으로 cell에 존재하는 차량에 대한 정보 취득
- 4차선 진입차도 4개에 각 100개의 cell로 이루어져 있어서, 총 1600개의 cell을 1초마다 갱신하여, 차량의 위치, 대기시간, 속도를 측정

| **차량 존재여부(queue_length)** | **차량 속도** | **차량 대기시간** |
|------------|------------|------------|
|![image](https://github.com/user-attachments/assets/2ea745cb-90e3-4ae2-80f0-677de5672ed9)|![image](https://github.com/user-attachments/assets/aba4525f-48a8-4a4f-846a-da87959f154d)|![image](https://github.com/user-attachments/assets/26103c13-e1ad-4a81-bec6-6c6d809e5251)|

## 3. 신호 선택하는 action space
![image](https://github.com/user-attachments/assets/9be5903d-9f9a-4e1a-9821-68c07f5356e1)
- 1 episode당 3600 steps으로 구성 (에피소드당 3600초)
- Green Light 종료후 3step 동안 동일한 방향의 Yellow Light 신호 실행 (ex. N/S Green Light 이후 다른 방향으로 신호가 바뀌면 신호가 바뀌기 전 N/S Yellow Light 신호 발생)
- 직진 및 우회전(2~4차선) 8 steps, 좌회전 (1차선) 4 steps로 신호 길이 고정
- Green Light 실행 이후 위의 4가지 방향 중 택 1 하여 실행

## 4. Reward
- 특정 방향의 신호만 계속해서 택하지 않도로 대기시간, 대기열에 공정성 지수를 적용 ([Jain's Fairness](https://en.wikipedia.org/wiki/Fairness_measure))

  $$
  R = -\left( w_1 \frac{W}{W_{\text{avg}}} + w_2 \frac{Q}{Q_{\text{avg}}} \right) + w_3 \left( \frac{w_1}{w_1 + w_2} F_w + \frac{w_2}{w_1 + w_2} F_q \right)
  $$  

- 전체 차선의 대기시간 총합, 대기열 총합을 줄일 수 있게 음의 보상값으로 설정
- 대기시간
  - 각 진입차도의 4개의 lane의 모든 차량 대기시간의 총합
  - 차선 내 존재하는 차량의 체류 기간(step)을 진입 시점-빠져나간 시점으로 계산
  - 에피소드 종료시 잔류 차량은 3600-진입step으로 계산
- 대기열
  - 각 차선에서 차량이 존재하고 속도도 0.1보다 작은 경우, 대기중인 차량으로 판단해 대기열로 count
  - 차선에 차량이 존재하지만, 속도가 0.1보다 빠르면 not count
  - 각 차선에 동일한 조건 적용해서 모든 대기열을 sum

## 5. History state 반영
![image](https://github.com/user-attachments/assets/2c2792ba-8068-4af5-a363-b0756a6deeb9)
- 이전 action이 끝난 후 state를 반영해서, DQN model input으로 넣음
- 세 state tensor를 concat 연산하여 입력

# 3. 실험 결과
## 1. History state 반영 전후 비교
![image](https://github.com/user-attachments/assets/9f417ccb-0944-47e3-9bd5-f0969accc552)

|-|prev_state+current_state|only current_state|
|:-:|:-:|:-:|
|wait time|28.81|31.23|
|queue length|0.294|0.396|
|reward|-0.042|-0.137|

이전 state를 반영한 결과가 더 좋게 나오는 것을 확인할 수 있다.
## 2. Fairness 유무 비교
![image](https://github.com/user-attachments/assets/0f0f890a-1022-42e0-8051-16a0f44c0bd3)

|-|fairness O|fairness X|
|:-:|:-:|:-:|
|wait time E |2.24|4.87|
|wait time W |4.13|10.32|
|wait time S |3.74|5.71|
|wait time N |3.31|3.49|

fairness를 반영한 결과 각 진입차선에서의 대기시간이 보다 더 균등하게 빠지는 것을 알 수 있다.

# 4. Contact
tngoc@naver.com
iankim010209@gmail.com
