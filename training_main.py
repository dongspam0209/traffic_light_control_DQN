from generator import CarGenerator
from util import set_sumo
import matplotlib.pyplot as plt
import traci

total_episode=100


def generate_car(seed):
    max_steps = 3600  # max step in one episode
    n_cars_generated = 1000  # number of car
    car_generator = CarGenerator(max_steps, n_cars_generated)
    car_generator.generate_car(seed)

def get_queue_length():
    halt_N=traci.edge.getLastStepHaltingNumber("N_in")
    halt_S=traci.edge.getLastStepHaltingNumber("S_in")
    halt_E=traci.edge.getLastStepHaltingNumber("E_in")
    halt_W=traci.edge.getLastStepHaltingNumber("W_in")
    total_queue_length=halt_N+halt_S+halt_E+halt_W

    return total_queue_length


if __name__ == "__main__":

    sumocfg_file_name = "cross.sumocfg"
    gui = True  # Change to False if you don't want the GUI
    max_steps=3600
    sumo_cmd = set_sumo(gui, sumocfg_file_name, max_steps)
    queue_length_store=[]

    episode=0


    while episode < total_episode:
        print(f'episode {episode}')
        generate_car(episode)
        

        traci.start(sumo_cmd)
    
        for step in range(max_steps):
            traci.simulationStep()  
            total_queue_length=get_queue_length()
        
           
        queue_length_store.append(total_queue_length)
        traci.close()  

        episode += 1

    episode_numbers=range(1,episode+1)
    plt.figure(figsize=(10,5))
    plt.plot(episode_numbers,queue_length_store,linestyle='-',color='b')
    plt.title('queue_length per episode')
    plt.xlabel('episode number')
    plt.ylabel('queue length')
    plt.xticks(episode_numbers)
    plt.grid(True)
    plt.show()
    # print(queue_length_store)