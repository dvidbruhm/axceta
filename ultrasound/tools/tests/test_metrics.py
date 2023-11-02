import pandas as pd
import numpy as np
import tools.compute_metrics as cm


def temp_to_sound_speed(temp_celsius: float) -> float:
    zero_c_kelvin = 273.15
    temp_kelvin = temp_celsius + zero_c_kelvin
    sound_speed = 20.02 * np.sqrt(temp_kelvin)
    return sound_speed


def tof_to_dist(tof, sound_speed):
    dist = sound_speed * tof * 10e-7 / 2
    return dist


if __name__ == "__main__":
    data_path = "data/test/input_data.csv"
    data = pd.read_csv(data_path, converters={})

    distances = list(data["distanceFromWeight"].values)
    loadcells = list(data["Loadcell_t"].values)
    fills = list(data["weightAlgo1FO_t"].values)

    output = cm.compute_metrics(loadcells, fills, distances)
    print(output)
