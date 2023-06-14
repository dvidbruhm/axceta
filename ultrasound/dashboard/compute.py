import tools.ultrasound_algos as algos
import json
import tools.utils as utils
from scipy import signal


raw_data_col = "rawdata"
temperature_col = "temperature"
pulse_count_col = "pulseCount"


def compute_wavefront(data, density, silo_metadata):
    wf_index = data.apply(lambda row: algos.wavefront(json.loads(row[raw_data_col]),
                                                      row[temperature_col], 0.5, 10, row[pulse_count_col]) * 2, axis=1)
    data["wf_index"] = wf_index
    wf_dist = data.apply(lambda row: utils.tof_to_dist(row["wf_index"], row[temperature_col]), axis=1)
    data["wf_dist"] = wf_dist
    wf_weight = data.apply(lambda row: utils.dist_to_volume_agco(row["wf_dist"], silo_metadata) * density / 100, axis=1)
    wf_perc = data.apply(lambda row: utils.dist_to_volume_agco(row["wf_dist"], silo_metadata) * 100 / silo_metadata["BinVolume"], axis=1)
    return wf_index, wf_weight, wf_perc


def compute_lowpass(data, density, silo_metadata):
    cutoff_freq = 150
    b, a = signal.butter(2, cutoff_freq / 250000, 'lowpass', analog=False)
    lowpass_index = data.apply(lambda row: algos.wavefront(signal.filtfilt(
        b, a, json.loads(row[raw_data_col])), row[temperature_col], 0.75, 10, row[pulse_count_col]) * 2, axis=1)
    data["lowpass_index"] = lowpass_index
    lowpass_dist = data.apply(lambda row: utils.tof_to_dist(row["lowpass_index"], row[temperature_col]), axis=1)
    data["lowpass_dist"] = lowpass_dist
    lowpass_weight = data.apply(lambda row: utils.dist_to_volume_agco(row["lowpass_dist"], silo_metadata) * density / 100, axis=1)
    lowpass_perc = data.apply(lambda row: utils.dist_to_volume_agco(row["lowpass_dist"], silo_metadata) * 100 / silo_metadata["BinVolume"], axis=1)
    return lowpass_index, lowpass_weight, lowpass_perc


def compute_cdm(data, density, silo_metadata):
    cdm_index = data.apply(lambda row: algos.center_of_mass(json.loads(row[raw_data_col]), row[pulse_count_col]) * 2, axis=1)
    data["cdm_index"] = cdm_index
    cdm_dist = data.apply(lambda row: utils.tof_to_dist(row["cdm_index"], row[temperature_col]), axis=1)
    data["cdm_dist"] = cdm_dist
    cdm_weight = data.apply(lambda row: utils.dist_to_volume_agco(row["cdm_dist"], silo_metadata) * density / 100, axis=1)
    cdm_perc = data.apply(lambda row: utils.dist_to_volume_agco(row["cdm_dist"], silo_metadata) / silo_metadata["BinVolume"] * 100, axis=1)
    return cdm_index, cdm_weight, cdm_perc


def compute_cdm_lowpass(data, density, silo_metadata):
    cutoff_freq = 150
    b, a = signal.butter(2, cutoff_freq / 250000, 'lowpass', analog=False)
    cdm_lowpass_index = data.apply(lambda row: algos.center_of_mass(signal.filtfilt(
        b, a, json.loads(row[raw_data_col])), row[pulse_count_col]) * 2, axis=1)
    data["cdm_lowpass_index"] = cdm_lowpass_index
    cdm_lowpass_dist = data.apply(lambda row: utils.tof_to_dist(row["cdm_lowpass_index"], row[temperature_col]), axis=1)
    data["cdm_lowpass_dist"] = cdm_lowpass_dist
    cdm_lowpass_weight = data.apply(lambda row: utils.dist_to_volume_agco(row["cdm_lowpass_dist"], silo_metadata) * density / 100, axis=1)
    cdm_lowpass_perc = data.apply(lambda row: utils.dist_to_volume_agco(
        row["cdm_lowpass_dist"], silo_metadata) / silo_metadata["BinVolume"] * 100, axis=1)
    return cdm_lowpass_index, cdm_lowpass_weight, cdm_lowpass_perc


def compute_best(data, density, silo_metadata):
    best_dist = []
    best_weight = []
    best_perc = []
    cutoff_freq = 150
    b, a = signal.butter(2, cutoff_freq / 250000, 'lowpass', analog=False)
    for i in range(len(data)):
        row = data.iloc[i]
        pulse = row[pulse_count_col]
        temperature = row[temperature_col]
        raw_data = json.loads(row[raw_data_col])

        lowpass = signal.filtfilt(b, a, raw_data)
        lowpass_bang_end = algos.detect_main_bang_end(lowpass, pulse)
        if max(raw_data[lowpass_bang_end:]) < 20:
            best_dist.append(utils.tof_to_dist(row["wf_index"], temperature))
            best_weight.append(row["wf_weight"])
            best_perc.append(row["wf_perc"])
        else:
            best_dist.append(utils.tof_to_dist(row["lowpass_index"], temperature))
            best_weight.append(row["lowpass_weight"])
            best_perc.append(row["lowpass_perc"])
    return best_dist, best_weight, best_perc
