import numpy as np
import matplotlib.pyplot as plt

# Weibull 분포를 따르는 난수 생성
y = np.random.weibull(2, 1000) * 1000

# x축으로 이동시킬 값
shift_value = 500

# 데이터를 x축으로 이동
y_shifted = y + shift_value

# 이동된 데이터의 히스토그램으로 분포 시각화
plt.hist(y_shifted, bins=50, alpha=0.75, color='blue', label='Weibull Distribution (Shifted)')

plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Weibull Distributed Data (Shifted)')
plt.legend()
plt.show()
