import math


def str_raw_data_to_list(raw_data, sep=","):
    return list(map(int, raw_data.strip("][").replace('"', "").split(sep)))


def dist_to_volume(dist, silo_data):
    sd = silo_data
    dist_offset = dist + sd["SensorOffset"]
    h_cone = sd["HeightGroundToTopOfCone"] - sd["HeightGroundToBin"]
    h_cylinder = sd["HeightGroundToTopOfCylinder"] - sd["HeightGroundToTopOfCone"]
    h_roof = sd["HeightGroundToTopOfBin"] - sd["HeightGroundToTopOfCylinder"]

    roof_volume = math.pi / 3 * h_roof * (sd["TopOfConeRadius"] ** 2 + sd["TopOfRoofRadius"] ** 2 + (sd["TopOfConeRadius"] * sd["TopOfRoofRadius"]))
    cylinder_volume = math.pi * sd["TopOfConeRadius"] ** 2 * h_cylinder
    hopper_volume = (math.pi / 3) * h_cone * (sd["TopOfConeRadius"] ** 2 + sd["BottomOfConeRadius"] ** 2 + sd["TopOfConeRadius"] * sd["BottomOfConeRadius"])
    total_volume = roof_volume + cylinder_volume + hopper_volume

    new_r1 = sd["TopOfConeRadius"] - (math.tan(math.radians(sd["ConeAngle"])) * (h_cone - (sd["HeightGroundToTopOfBin"] - sd["HeightGroundToBin"] - dist_offset)))

    volume_in_cylinder = math.pi * (sd["TopOfConeRadius"] ** 2) * ((h_cylinder + h_roof) - dist_offset) + hopper_volume

    volume_in_cone = (math.pi / 3) * (new_r1**2 + sd["BottomOfConeRadius"] ** 2 + new_r1 * sd["BottomOfConeRadius"]) * (sd["HeightGroundToTopOfBin"] - sd["HeightGroundToBin"] - dist_offset)

    new_r3 = dist_offset * math.tan(math.radians(sd["RoofAngle"])) + sd["TopOfRoofRadius"]
    volume_in_roof = (math.pi / 3) * (new_r3**2 + sd["TopOfConeRadius"] ** 2 + new_r3 * sd["TopOfConeRadius"]) * (h_roof - dist_offset)

    current_volume = None
    if dist_offset < h_roof:
        current_volume = volume_in_roof + cylinder_volume + hopper_volume
    elif dist_offset >= (h_roof + h_cylinder):
        current_volume = volume_in_cone
    else:
        current_volume = volume_in_cylinder

    return current_volume


def tof_to_dist(tof, temp_celsius, frequency=20000):
    zero_c_kelvin = 273.15
    temp_kelvin = temp_celsius + zero_c_kelvin
    sound_speed = 20.02 * math.sqrt(temp_kelvin)
    dist = sound_speed * tof * 1e-6 / (frequency / 1000000)
    return dist
