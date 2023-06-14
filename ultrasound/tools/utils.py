import math
import numpy as np

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


def dist_to_volume(dist: float, silo_name: str, silo_data: dict) -> np.ndarray:
    # print(conversion_data.columns)
    # silo_data = conversion_data.loc[conversion_data["LocationName"] == silo_name]

    # print(silo_data)
    # print(silo_name)
    # print(conversion_data)
    h1, h2, h3 = silo_data["H1"], silo_data["H2"], silo_data["H3"]
    # print("H1, H2, H3 : ", h1, h2, h3)
    diam, angle = silo_data["Diametre"], silo_data["Angle (degré)"]
    offset = silo_data["Offset du device"]
    # print("Diameter, angle, offset : ", diam, angle, offset)
    max_d = h3 - h1 - offset
    r1 = diam / 2
    cone_height = h2 - h1
    r2 = r1 - (cone_height * math.tan(math.radians(angle)))
    if "CDPQA" in silo_name:
        r2 = 0.4445
    # print("R1, R2, Cone Height : ", r1, r2, cone_height)
    vol_cone = (1 / 3) * math.pi * (r1**2 + r2**2 + r1 * r2) * cone_height
    h_cyl = h3 - h2
    vol_cyl = math.pi * r1**2 * h_cyl
    # print("VolumeCone, HeightCylinder, VolumeCylinder : ", vol_cone, h_cyl, vol_cyl)
    vol_tot = vol_cyl + vol_cone
    dist_tot = dist + offset
    new_r1 = (cone_height - (dist_tot - h_cyl)) * math.tan(math.radians(angle)) + r2 if dist_tot > h_cyl else 0
    # print("Total Volume, Total distance, New R1 : ", vol_tot, dist_tot, new_r1)
    density = silo_data["densité de moulé (kg/hl)"]
    if dist_tot <= h_cyl:
        vol_hecto = (math.pi * r1 * r1 * (h_cyl - dist_tot) + vol_cone) * 10
    else:
        vol_hecto = (1 / 3) * math.pi * (new_r1**2 + r2**2 + new_r1 * r2) * (h3 - h1 - dist_tot) * 10

    perc_fill = vol_hecto / vol_tot * 10
    # print("Volume Hecto, Percent : ", vol_hecto, perc_fill)

    # print(vol_tot, dist_tot, new_r1)
    # print(density, vol_hecto, perc_fill)

    weight = vol_hecto * density / 1000
    return weight


def temp_to_sound_speed(temp_celsius: float) -> float:
    zero_c_kelvin = 273.15
    temp_kelvin = temp_celsius + zero_c_kelvin
    sound_speed = 20.02 * np.sqrt(temp_kelvin)
    return sound_speed


def dist_to_tof(dist, temp_celsius):
    sound_speed = temp_to_sound_speed(temp_celsius)
    tof = dist * 2 / (sound_speed * 1e-6)
    return tof


def tof_to_dist(tof, temp_celsius):
    zero_c_kelvin = 273.15
    temp_kelvin = temp_celsius + zero_c_kelvin
    sound_speed = 20.02 * np.sqrt(temp_kelvin)
    dist = sound_speed * tof * 1e-6 / 2
    return dist


def cotan(angle):
    return 1 / math.tan(angle)


def weight_to_tof(weight, silo_data, density, temp_celsius):
    volume = weight * 100 / density
    if volume <= silo_data["ConeVolume"]:
        print(1)
        offset = silo_data["CylinderHeight"] + silo_data["ConeHeight"] - silo_data["SensorOffset"]
        dist = offset - (((volume / silo_data["ConeVolume"] * ((silo_data["TopOfConeRadius"] ** 3) - (silo_data["BottomOfConeRadius"] ** 3)
                                                               ) + silo_data["BottomOfConeRadius"] ** 3) ** 0.333333333) - silo_data["BottomOfConeRadius"]) * cotan(math.radians(silo_data["ConeAngle"]))
    else:
        print(2)
        dist = -(volume - silo_data["ConeVolume"]) / (math.pi * (silo_data["TopOfConeRadius"] ** 2)
                                                      ) + silo_data["CylinderHeight"] - silo_data["SensorOffset"]

    print(dist)
    tof = dist_to_tof(dist, temp_celsius)
    print(tof)
    return tof


weight_to_tof(30, agco_silo_data, 65, 0)
