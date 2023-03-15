from pathlib import Path
from rich import print
import numpy as np


def assert_csv(file: Path):
    assert str(file.absolute())[-4:] == ".csv", "Input file should be a csv"


def temp_to_sound_speed(temp_celsius: float) -> float:
    zero_c_kelvin = 273.15
    temp_kelvin = temp_celsius + zero_c_kelvin
    sound_speed = 20.02 * np.sqrt(temp_kelvin)
    return sound_speed


def tof_to_dist(tof, sound_speed):
    dist = sound_speed * tof * 1e-6 / 2
    return dist


def tof_to_dist2(tof, temp_celsius):
    sound_speed = temp_to_sound_speed(temp_celsius)
    dist = tof_to_dist(tof, sound_speed)
    return dist
