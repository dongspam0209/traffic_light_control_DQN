import traci
import numpy as np
import pandas as pd

# action phase 정의

PHASE_NS_GREEN = 0  # action 0 code 00 -> 북/남 직진
PHASE_NS_YELLOW = 1 
PHASE_NSL_GREEN = 2  # action 1 code 01 -> 북/남 좌회전
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action 2 code 10 -> 동/서 직진
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action 3 code 11 -> 동/서 좌회전
PHASE_EWL_YELLOW = 7


class Simulation:
    def __init__(self,Cargenerator,sumo_cmd,max_steps,num_states):
        self._Cargenerator=Cargenerator
        self._sumo_cmd=sumo_cmd
        self._max_steps=max_steps
        self._num_states=num_states

    def run(self,episode):
        lane=["W_in_0","W_in_1","W_in_2","W_in_3","N_in_0","N_in_1","N_in_2","N_in_3",
            "E_in_0","E_in_1","E_in_2","E_in_3","S_in_0","S_in_1","S_in_2","S_in_3"]
        self._Cargenerator.generate_car(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating")
        for step in range(self._max_steps):
            current_state=self._get_state()
            df = pd.DataFrame(current_state,index=lane)
            df.to_csv('./intersection/get_state.csv', index=True)
            traci.simulationStep()

        
        traci.close()
    
    def _get_state(self):
        lane=["W_in_0","W_in_1","W_in_2","W_in_3","N_in_0","N_in_1","N_in_2","N_in_3",
              "E_in_0","E_in_1","E_in_2","E_in_3","S_in_0","S_in_1","S_in_2","S_in_3"]
        state=np.zeros((16,100))
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

            state[lane_group][lane_cell]=1 # cell occupied

        
        
        return state.tolist()




        