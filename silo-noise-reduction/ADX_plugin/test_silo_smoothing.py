import numpy as np
import matplotlib.pyplot as plt

from silo_smoothing import savgol
from scipy.signal import savgol_filter


test_data = np.array(np.sin(np.arange(0, 10, 0.1)))
test_data = test_data + np.random.uniform(0, 1, test_data.shape) + 2
print(test_data.shape)
filtered_data = []
for i in range(len(test_data)):
    filt_v = savgol(test_data[max(0, i-10):i+1], 11, 2, 3)
    filtered_data.append(filt_v)

plt.plot(test_data)
plt.plot(filtered_data)
plt.plot(savgol_filter(test_data, 11, 2))
plt.show()
