from generator import CarGenerator
from util import set_sumo, set_train_path
from simulation import Simulation
import matplotlib.pyplot as plt
from Model import DQN
import traci
from replay import ReplayMemory
import wandb

from visualization import Visualization

################################################################
total_episode = 500
n_cars_generated = 1000

num_states = (3, 16, 100)
num_actions = 4

yellow_duration = 3
green_duration = 8
green_turn_duration = 4
memory_capacity = 1000

wandb.init(
    # set the wandb project where this run will be logged
    project="tlsc-project",
    # track hyperparameters and run metadata
    config={
        "learning_rate": 0.0001,
        "architecture": "DQN",
        "dataset": "CIFAR-100",
        "epochs": 10000,
    }
)
################################################################

if __name__ == "__main__":

    #################################################################
    sumocfg_file_name = "cross.sumocfg"
    gui = False  # Change to False if you don't want the GUI
    max_steps = 3600
    sumo_cmd = set_sumo(gui, sumocfg_file_name, max_steps)
    path = set_train_path('plot')
    ##################################################################

    ReplayMemory = ReplayMemory(
        memory_capacity
    )
    CarGenerator = CarGenerator(
        max_steps,
        n_cars_generated
    )

    Simulation = Simulation(
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
    DQN = DQN(
        num_states,
        num_actions
    )
    Visualization = Visualization(
        path,
        dpi=96
    )

    episode = 0

    epsilon = 1.0
    min_epsilon = 0.1
    decay_rate = 0.99

    while episode < total_episode:
        print(f'episode {episode}')
        epsilon = max(min_epsilon, epsilon * decay_rate)
        Simulation.run(episode, epsilon)
        print(f'queue length in epsiode {episode}', Simulation.queue_length_store[episode])
        print(f'loss in epsiode {episode}', Simulation.loss_store[episode])
        print(f'wait time in epsiode {episode}', Simulation.wait_time_store[episode])
        print(f'reward in epsiode {episode}', Simulation.reward_store[episode])

        # wandb
        wandb.log({
            "episode": episode,
            "epsilon": epsilon,
            "queue length": Simulation.queue_length_store[episode],
            "loss": Simulation.loss_store[episode],
            "wait time": Simulation.wait_time_store[episode],
            "reward": Simulation.reward_store[episode],
        })

        episode += 1

    