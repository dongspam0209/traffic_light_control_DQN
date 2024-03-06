import numpy as np
import matplotlib.pyplot as plt
mean_1=np.random.uniform(0,0.5)
mean_2=np.random.uniform(0.5,1.0)
mean_3=np.random.uniform(1.0,1.5)
mean_4=np.random.uniform(1.5,2.0)
sigma_1=0.2
sigma_2=0.1

test_1=np.random.randint(0,3400,size=150)
test_2=np.random.randint(0,3400,size=150)
test_3=np.random.randint(0,3400,size=150)
test_4=np.random.randint(0,3400,size=150)

dist_1=np.random.lognormal(mean_1,sigma_1,100)*400
dist_2=np.random.lognormal(mean_2,sigma_1,100)*400
dist_3=np.random.lognormal(mean_3,sigma_2,100)*400
dist_4=np.random.lognormal(mean_4,sigma_2,100)*400


dist_1=np.rint(dist_1)
dist_2=np.rint(dist_2)
dist_3=np.rint(dist_3)
dist_4=np.rint(dist_4)

result_1=np.append(test_1,dist_1)
result_2=np.append(test_2,dist_2)
result_3=np.append(test_3,dist_3)
result_4=np.append(test_4,dist_4)

# plt.hist(result_1, bins=50, alpha=0.6, label='incoming road_1')
# plt.hist(result_2, bins=50, alpha=0.6, label='incoming road_2')
# plt.hist(result_3, bins=50, alpha=0.6, label='incoming road_3')
plt.hist(result_4, bins=50, alpha=0.6, label='incoming road_4')

plt.title('Log-normal Distributions')
plt.xlabel('Time')
plt.ylabel('Car_n_generated')
plt.legend()
plt.show()