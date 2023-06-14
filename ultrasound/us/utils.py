from pathlib import Path
from rich import print
import numpy as np
import math


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


def dist_to_tof(dist, temp_celsius):
    sound_speed = temp_to_sound_speed(temp_celsius)
    tof = dist * 2 / (sound_speed * 1e-6)
    return tof


agco_silo_data = {
    "ConeHeight": 2.918,
    "HeightGroundToBin": 0.7,
    "HeightGroundToTopOfCone": 3.618,
    "HeightGroundToTopOfCylinder": 6.895,
    "HeightGroundToTopOfBin": 8.230,
    "CylinderHeight": 3.277,
    "ConeAngle": 30,
    "RoofAngle": 50,
    "TopOfConeRadius": 1.82,
    "BottomOfConeRadius": 0.135,
    "TopOfRoofRadius": 0.229,
    "ConeVolume": 10.93,
    "BinVolume": 50.318,
    "SensorOffset": 0.335,
}

"""
let SensorOffset = toreal(siloDetails["SensorOffset"])
let H_Cone = toreal(siloDetails["ConeHeight"])
let _Height1 = toreal(siloDetails["HeightGroundToTopOfBin"])
let _Height2 = toreal(siloDetails["HeightGroundToTopOfCone"])
let _Height3 = toreal(siloDetails["HeightGroundToBin"])
let H_Cylindre = toreal(siloDetails["CylinderHeight"])
let Angle = toreal(siloDetails["ConeAngle"])
let R1TopCone = toreal(siloDetails["TopOfConeRadius"])
let R2BasCone = toreal(siloDetails["BottomOfConeRadius"])
let VolumeDuCone = toreal(siloDetails["ConeVolume"])
let VolumeTotal = toreal(siloDetails["BinVolume"])
let FeedRemainingM3 = LoadCellWeight_t * 1000 / density / 10.0
iff(FeedRemainingM3 <= VolumeDuCone, H_Cylindre + H_Cone - SensorOffset -
    (pow((FeedRemainingM3 / VolumeDuCone * (pow(R1TopCone, 3) - pow(R2BasCone, 3)) + pow(R2BasCone, 3)),
         0.3333333333333333) - R2BasCone) * cot(radians(Angle)),
    -(FeedRemainingM3 - VolumeDuCone) / (pi() * pow(R1TopCone, 2)) + H_Cylindre - SensorOffset)
"""


def cotan(angle):
    return 1 / math.tan(angle)


def weight_to_tof(weight, silo_data, density, temp_celsius):
    volume = weight * 100 / density
    if volume <= silo_data["ConeVolume"]:
        offset = silo_data["CylinderHeight"] + silo_data["CylinderHeight"] - silo_data["SensorOffset"]
        dist = offset - (((volume / silo_data["ConeVolume"] * (silo_data["TopOfConeRadius"] ** 3 - silo_data["BottomOfConeRadius"] **
                         3) + silo_data["BottomOfConeRadius"] ** 3) ** 0.333333333) - silo_data["BottomOfConeRadius"]) * cotan(math.radians(silo_data["ConeAngle"]))
    else:
        dist = -(volume - silo_data["ConeVolume"]) / (math.pi * silo_data["TopOfConeRadius"]
                                                      ** 2) + silo_data["CylinderHeight"] - silo_data["SensorOffset"]

    tof = dist_to_tof(dist, temp_celsius)
    return tof


def dist_to_volume_agco(dist, silo_data):
    sd = silo_data
    dist_offset = dist + sd["SensorOffset"]
    h_cone = sd["HeightGroundToTopOfCone"] - sd["HeightGroundToBin"]
    h_cylinder = sd["HeightGroundToTopOfCylinder"] - sd["HeightGroundToTopOfCone"]
    h_roof = sd["HeightGroundToTopOfBin"] - sd["HeightGroundToTopOfCylinder"]

    roof_volume = math.pi / 3 * h_roof * (sd["TopOfConeRadius"] ** 2 + sd["TopOfRoofRadius"] ** 2 + (sd["TopOfConeRadius"] * sd["TopOfRoofRadius"]))
    cylinder_volume = math.pi * sd["TopOfConeRadius"] ** 2 * h_cylinder
    hopper_volume = (math.pi / 3) * h_cone * (sd["TopOfConeRadius"] ** 2 + sd["BottomOfConeRadius"] ** 2
                                              + sd["TopOfConeRadius"] * sd["BottomOfConeRadius"])
    total_volume = roof_volume + cylinder_volume + hopper_volume

    new_r1 = sd["TopOfConeRadius"] - (math.tan(math.radians(sd["ConeAngle"])) *
                                      (h_cone - (sd["HeightGroundToTopOfBin"] - sd["HeightGroundToBin"] - dist_offset)))

    volume_in_cylinder = math.pi * (sd["TopOfConeRadius"] ** 2) * ((h_cylinder + h_roof) - dist_offset) + hopper_volume

    volume_in_cone = (math.pi / 3) * (new_r1 ** 2 + sd["BottomOfConeRadius"] ** 2
                                      + new_r1 * sd["BottomOfConeRadius"]) * (sd["HeightGroundToTopOfBin"] - sd["HeightGroundToBin"] - dist_offset)

    new_r3 = dist_offset * math.tan(math.radians(sd["RoofAngle"])) + sd["TopOfRoofRadius"]
    volume_in_roof = (math.pi / 3) * (new_r3 ** 2 + sd["TopOfConeRadius"] ** 2 + new_r3 * sd["TopOfConeRadius"]) * (h_roof - dist_offset)

    current_volume = None
    if dist_offset < h_roof:
        current_volume = volume_in_roof + cylinder_volume + hopper_volume
    elif dist_offset >= (h_roof + h_cylinder):
        current_volume = volume_in_cone
    else:
        current_volume = volume_in_cylinder

    return current_volume


def dist_to_volume(dist: float, silo_data: dict) -> np.ndarray:
    h1, h2, h3 = silo_data["H1"], silo_data["H2"], silo_data["H3"]
    diam, angle = silo_data["Diametre"], silo_data["Angle (degré)"]
    offset = silo_data["Offset du device"]
    max_d = h3 - h1 - offset
    r1 = diam / 2
    cone_height = h2 - h1
    r2 = r1 - (cone_height * math.tan(math.radians(angle)))
    vol_cone = (1 / 3) * math.pi * (r1**2 + r2**2 + r1 * r2) * cone_height
    h_cyl = h3 - h2
    vol_cyl = math.pi * r1**2 * h_cyl
    vol_tot = vol_cyl + vol_cone
    dist_tot = dist + offset
    new_r1 = (cone_height - (dist_tot - h_cyl)) * math.tan(math.radians(angle)) + r2 if dist_tot > h_cyl else 0
    density = silo_data["densité de moulé (kg/hl)"]
    if dist_tot <= h_cyl:
        vol_hecto = (math.pi * r1 * r1 * (h_cyl - dist_tot) + vol_cone) * 10
    else:
        vol_hecto = (1 / 3) * math.pi * (new_r1**2 + r2**2 + new_r1 * r2) * (h3 - h1 - dist_tot) * 10

    perc_fill = vol_hecto / vol_tot * 10
    weight = vol_hecto * density / 1000
    return weight
