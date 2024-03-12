import traci
import numpy as np
import pandas as pd

class Simulation:
    def __init__(self,Cargenerator,sumo_cmd,max_steps,num_states):
        self._Cargenerator=Cargenerator
        self._sumo_cmd=sumo_cmd
        self._max_steps=max_steps
        self._num_states=num_states

    # def save_state_to_csv(state, lanes, base_filename='state'):
    #     for i in range(state[0]):
    #         df = pd.DataFrame(state[i])
    #         df.index=lanes
    #         filename = f'./intersection/{base_filename}_{i}.csv'
    #         df.to_csv(filename)
    #         print(f'Saved dimension {i} to {filename}')
    
    def run(self,episode):
        lane=["W_in_0","W_in_1","W_in_2","W_in_3","N_in_0","N_in_1","N_in_2","N_in_3",
            "E_in_0","E_in_1","E_in_2","E_in_3","S_in_0","S_in_1","S_in_2","S_in_3"]
        self._Cargenerator.generate_car(seed=episode) # car generation
        traci.start(self._sumo_cmd)
        print("Simulating")
        for step in range(self._max_steps):
            current_state=self._get_state()

            # self.save_state_to_csv(current_state,lane)
            df1=pd.DataFrame(current_state[0],index=lane)
            df1.to_csv('./intersection/generate_exist.csv')
            df2=pd.DataFrame(current_state[1],index=lane)
            df2.to_csv('./intersection/generate_velocity.csv')
            df3=pd.DataFrame(current_state[2],index=lane)
            df3.to_csv('./intersection/generate_waiting_time.csv')

            traci.simulationStep()

        
        traci.close()
    
    def _get_state(self):
        lane=["W_in_0","W_in_1","W_in_2","W_in_3","N_in_0","N_in_1","N_in_2","N_in_3",
              "E_in_0","E_in_1","E_in_2","E_in_3","S_in_0","S_in_1","S_in_2","S_in_3"]
        state=np.zeros((3,16,100))
        vehicle_list=traci.vehicle.getIDList()
        lane_group=0

        for veh_id in vehicle_list:
            lane_position = traci.vehicle.getLanePosition(veh_id) # car position in lane
            lane_id=traci.vehicle.getLaneID(veh_id) # -> lane
            lane_position=700-lane_position # traffic light ~> max len of road 0~700
            lane_cell= int(lane_position/7) # 7m : hegiht of the cell

            for idx in range(len(lane)):
                if lane_id==lane[idx]:
                    lane_group=idx

            state[0][lane_group][lane_cell]=1 # cell occupied
            state[1][lane_group][lane_cell]=traci.vehicle.getSpeed(veh_id) # vehicle velocity
            state[2][lane_group][lane_cell]=traci.vehicle.getAccumulatedWaitingTime(veh_id) # waiting time

        
        
        return state.tolist()




        