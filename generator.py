
import numpy as np
import math

class CarGenerator:
    def __init__(self, max_steps, n_cars_generated):
        self._n_cars_generated = n_cars_generated  # 한 에피소드당 생성할 차량 수
        self._max_steps = max_steps  # 최대 시뮬레이션 스텝

    def generate_car(self, seed):
        np.random.seed(seed)
        
        # car information list
        vehicles_info = []

        # car route (start_to_end)
        route_ids_straight = ["W_to_E", "N_to_S", "E_to_W", "S_to_N"]
        route_ids_turn=["W_to_N", "N_to_E" , "E_to_S" , "S_to_W","W_to_S","N_to_W", "E_to_N", "S_to_E"]
        # car generation distribution : normal distribution selection
        # only one way is selected, that way follows normal distribution to describe commuting time
        distribution_switch=np.random.uniform()

        if(distribution_switch<0.25):
            weibull_distribution_route=["W_to_E", "W_to_S","W_to_N"]
        elif distribution_switch<0.5 and distribution_switch>=0.25 :
            weibull_distribution_route=["E_to_W", "E_to_S","E_to_N"]
        elif distribution_switch<0.75 and distribution_switch>=0.5 :
            weibull_distribution_route=["N_to_W", "N_to_S","N_to_E"]
        elif distribution_switch>=0.75 :
            weibull_distribution_route=["S_to_W", "S_to_N","S_to_E"]

        print('weibull distribution car lane is',weibull_distribution_route)
        # limit of car generation timing 0~3000

        lambda_=700

        for i in range(self._n_cars_generated):
            car_id = f"vehicle_{i}"
            ### straight or left switch ###
            straight_or_left_switch=np.random.uniform()
            if(straight_or_left_switch<0.75):
                route_id=np.random.choice(route_ids_straight)
            else:
                route_id=np.random.choice(route_ids_turn)
            ################################
                

            # selected lane follows weibull distribution
            if route_id in weibull_distribution_route: # traffic jam lane
                depart=np.random.weibull(2)*lambda_
            else:
                # depart=np.random.uniform(0, 3000)
                depart=np.random.weibull(2)*lambda_

            # others are just uniform distribution
            depart = np.rint(depart)
            depart = np.clip(depart, 0, 3000)
            vehicles_info.append((depart, car_id, route_id))

        # sort depart time
        vehicles_info.sort()

        # .rou.xml update
        with open("./intersection/cross.rou.xml", "w") as routes:
            print("""                
<routes>
            <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" laneChangeMode="5242898"/>
            
            <!-- route definition (the car route) -->

            <route id="W_to_N" edges="W_in N_out"/>
            <route id="W_to_E" edges="W_in E_out"/>
            <route id="W_to_S" edges="W_in S_out"/>
            <route id="N_to_W" edges="N_in W_out"/>
            <route id="N_to_E" edges="N_in E_out"/>
            <route id="N_to_S" edges="N_in S_out"/>
            <route id="E_to_W" edges="E_in W_out"/>
            <route id="E_to_N" edges="E_in N_out"/>
            <route id="E_to_S" edges="E_in S_out"/>
            <route id="S_to_W" edges="S_in W_out"/>
            <route id="S_to_N" edges="S_in N_out"/>
            <route id="S_to_E" edges="S_in E_out"/>""",file=routes)

            for depart, car_id, route_id in vehicles_info:
                print(f'    <vehicle id="{car_id}" type="standard_car" route="{route_id}" depart="{depart}" departLane="random" departSpeed="10" laneChangeMode="5242898" />' , file=routes)

            print("</routes>", file=routes)


