from generator import CarGenerator
from util import set_sumo
from simulation import Simulation
import matplotlib.pyplot as plt
import traci

################################################################
total_episode=100
max_steps=3600
n_cars_generated=1000
num_states=1600
################################################################

if __name__ == "__main__":

#################################################################
    sumocfg_file_name = "cross.sumocfg"
    gui = True  # Change to False if you don't want the GUI
    max_steps=3600
    sumo_cmd = set_sumo(gui, sumocfg_file_name, max_steps)
##################################################################
    
    CarGenerator=CarGenerator(
        max_steps,
        n_cars_generated
    )

    Simulation=Simulation(
        CarGenerator,
        sumo_cmd,
        max_steps,
        num_states
    )
    episode=0


    while episode < total_episode:
        print(f'episode {episode}')

        Simulation.run(episode)

        episode += 1
