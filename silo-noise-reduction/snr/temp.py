from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ADX_plugin.silo_smoothing as sm

data = pd.DataFrame(pd.read_csv(Path("data", "Avinor1485B.csv")))
print(data)
print(data.columns)
data["DistanceFO"] /= 1000.0
data["DistanceCDM"] /= 1000.0

data["WeightCDM"] = [1 if d > 2.0 else 0 for d in data["DistanceFO"]]
data["WeightedVolume"] = (data["UsedVolumeCDM"] * data["WeightCDM"]) + data["UsedVolumeFO"] * (1 - data["WeightCDM"])
print(data["WeightCDM"])
vols = data["WeightedVolume"].values
new_vols = np.zeros_like(vols)
for i in range(1, len(vols)):
    v = vols[i]
    prev_v = vols[i-1]
    if v - prev_v > 0.05:
        new_vols[i] = prev_v
    else:
        new_vols[i] = v
data["volume2"] = new_vols
smoothed_values = np.zeros_like(data["WeightedVolume"].values)
values_to_smooth = data["volume2"].values
window, order, offset, fixed_window = 15, 2, 3, False
for i in range(len(values_to_smooth)):
    smoothed_values[i], _ = sm.savgol(values_to_smooth[max(0, i-window+1):i+1], window, order, offset, fixed_window)
data["SmoothedUsedVolume"] = smoothed_values


plt.plot(data["WeightedVolume"], label="volume")
#plt.plot(data["volume2"], label="new volume")
plt.plot(data["SmoothedUsedVolume"], label="s volume")
plt.legend(loc="best")
plt.show()

data = data.drop(columns=["volume2"])
data.to_csv(Path("data/Avinor-1485B-smoothed.csv"))
