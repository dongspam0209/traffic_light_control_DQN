import numpy as np
state_history= np.zeros((3,3,16,100))
state_history=state_history.tolist()
print(len(state_history))
state_history.pop(0)
print(len(state_history))