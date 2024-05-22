import pandas as pd

df1=pd.read_csv('generate_exist.csv',encoding='utf-8')
df2=pd.read_csv('generate_velocity.csv',encoding='utf-8')

df1=df1.transpose()
df2=df2.transpose()
df1.columns=df1.iloc[0]
df2.columns=df2.iloc[0]
df1=df1[1:]
df2=df2[1:]

# print(df1)
# print(df2)
queue_length_sum_per_lane_list=[]
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

queue_len_per_lane=queue_length_sum_per_lane_list
reward_queue_length=sum(queue_length_sum_per_lane_list)

print(queue_len_per_lane)
print(reward_queue_length)