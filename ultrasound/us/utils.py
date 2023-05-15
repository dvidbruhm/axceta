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


"""
let metadata = dynamic({"ConeHeight": 2.918,
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
    "SensorOffset": 0.335});
let RoofVolume = pi() / 3 * (pow(R1TopCone, 2) + pow(R3TopOfRoof, 2) + R3TopOfRoof * R1TopCone) * (h_Roof);
let CylinderVolume = pi() * pow(R1TopCone, 2) * H_Cylindre;
let HopperVolume = pi() / 3 * (pow(R1TopCone, 2) + pow(R2BasCone, 2) + R2BasCone * R1TopCone) * (_Height2 - _Height1);
let VolumeTotal = RoofVolume + CylinderVolume + HopperVolume;
let newR1= R1TopCone - (tan(radians(HopperAngle)) * (H_Cone - (_Height4 - _Height1 - Distance_m)));
let volumeInCylinder=(pi() * pow(R1TopCone, 2) * ((H_Cylindre + h_Roof) - Distance_m) + HopperVolume);
let volumeInCone= pi() / 3 * (pow(newR1, 2) + pow(R2BasCone, 2) + R2BasCone * newR1) * (_Height4 - _Height1 - Distance_m);
let newR3=(Distance_m) * tan(radians(RoofAngle)) + R3TopOfRoof;
let volumeInRoof = pi() / 3 * (pow(newR3, 2) + pow(R1TopCone, 2) + R1TopCone * newR3) * (h_Roof - Distance_m);
"""


def dist_to_volume_agco(dist, silo_data):
    sd = silo_data
    h_roof = sd["HeightGroundToTopOfBin"] - sd["HeightGroundToTopOfCylinder"]
    h_cylinder = sd["HeightGroundToTopOfCylinder"] - sd["HeightGroundToTopOfCone"]
    h_cone = sd["HeightGroundToTopOfCone"] - sd["HeightGroundToBin"]
    roof_volume = (math.pi / 3) * h_roof * (sd["TopOfConeRadius"] ** 2 + sd["TopOfRoofRadius"] ** 2 + sd["TopOfConeRadius"] * sd["TopOfRoofRadius"])
    cylinder_volume = math.pi * sd["TopOfConeRadius"] ** 2 * h_cylinder
    hopper_volume = (math.pi / 3) * h_cone * (sd["TopOfConeRadius"] ** 2 + sd["BottomOfConeRadius"] ** 2
                                              + sd["TopOfConeRadius"] * sd["BottomOfConeRadius"])
    total_volume = roof_volume + cylinder_volume + hopper_volume

    print(hopper_volume)
    new_r1 = sd["TopOfConeRadius"] - (math.tan(math.radians(sd["ConeAngle"])) *
                                      (h_cone - (sd["HeightGroundToTopOfBin"] - sd["HeightGroundToBin"] - dist)))
    volume_in_cylinder = math.pi * sd["TopOfConeRadius"] ** 2 * (h_cylinder + h_cone - dist) + hopper_volume
    volume_in_cone = (math.pi / 3) * (new_r1 ** 2 + sd["BottomOfConeRadius"] ** 2
                                      + new_r1 * sd["BottomOfConeRadius"]) * (sd["HeightGroundToTopOfBin"] - sd["HeightGroundToBin"] - dist)

    new_r3 = dist * math.tan(math.radians(sd["RoofAngle"])) + sd["TopOfRoofRadius"]
    volume_in_roof = (math.pi / 3) * (new_r3 ** 2 + sd["TopOfConeRadius"] ** 2 + new_r3 * sd["TopOfConeRadius"]) * (h_roof - dist)

    current_volume = None
    if dist < h_roof:
        print("1")
        current_volume = volume_in_roof + cylinder_volume + hopper_volume
    elif dist >= (h_roof + h_cylinder):
        print("2")
        current_volume = volume_in_cone
    else:
        print("3")
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
