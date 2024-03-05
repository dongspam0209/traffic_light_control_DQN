from generator import CarGenerator
def generate_car(seed):
    max_steps = 3600  # max step in one episode
    n_cars_generated = 1000  # number of car
    car_generator = CarGenerator(max_steps, n_cars_generated)
    car_generator.generate_car(seed)
if __name__=="__main__":
    episode=0

    while episode <1000:
       generate_car(episode)
       episode+=1