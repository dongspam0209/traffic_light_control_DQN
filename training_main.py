from generator import CarGenerator
from util import set_sumo , set_train_path
from simulation import Simulation
import matplotlib.pyplot as plt
from environment.traffic_signal import TrafficSignal
from Model import DQN
import traci
from replay import ReplayMemory

from visualization import Visualization
################################################################
total_episode=30
n_cars_generated=1000

num_states=(3,16,100)
num_actions=4

yellow_duration=3
green_duration=8
green_turn_duration=4
memory_capacity=1000


################################################################

if __name__ == "__main__":

#################################################################
    sumocfg_file_name = "cross.sumocfg"
    gui = False  # Change to False if you don't want the GUI
    max_steps=3600
    sumo_cmd = set_sumo(gui, sumocfg_file_name, max_steps)
    path=set_train_path('plot')
##################################################################


    ReplayMemory=ReplayMemory(
        memory_capacity
    )
    CarGenerator=CarGenerator(
        max_steps,
        n_cars_generated
    )

    Simulation=Simulation(
        DQN,
        ReplayMemory,
        CarGenerator,
        sumo_cmd,
        max_steps,
        num_states,
        num_actions,
        green_duration,
        yellow_duration,
        green_turn_duration,
        
    )
    DQN=DQN(
        num_states,
        num_actions
    )
    Visualization=Visualization(
        path,
        dpi=96
    )
    episode=0


    while episode < total_episode:
        print(f'episode {episode}')
        epsilon=1.0-(episode/total_episode)
        Simulation.run(episode,epsilon)
        episode += 1

    Visualization.save_data_and_plot(data=Simulation.queue_length_store,filename='queue',xlabel='Episode',ylabel='queue length')
    Visualization.save_data_and_plot(data=Simulation.loss_store,filename='loss',xlabel='Episode',ylabel='loss')


 