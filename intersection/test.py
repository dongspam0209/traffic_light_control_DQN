import pandas as pd
import numpy as np

lane=["W_in_0","W_in_1","W_in_2","W_in_3","N_in_0","N_in_1","N_in_2","N_in_3",
    "E_in_0","E_in_1","E_in_2","E_in_3","S_in_0","S_in_1","S_in_2","S_in_3"]

df1=pd.read_csv('generate_exist.csv').transpose()
df2=pd.read_csv('generate_velocity.csv').transpose()

car_presence=np.zeros((16,100))
car_presence += (df2[0]>0)

car_presence_boolean=car_presence>0

car_presence_df=pd.DataFrame(car_presence_boolean,index=lane).transpose()


halted_vehicles_per_lane = []

for direction in ['W_in_', 'S_in_', 'N_in_', 'E_in_']:
        halted_count = ((df2.filter(like=direction) <= 0.1) & df1.filter(like=direction)).sum().sum()
        halted_vehicles_per_lane.append(halted_count)


print(halted_vehicles_per_lane)