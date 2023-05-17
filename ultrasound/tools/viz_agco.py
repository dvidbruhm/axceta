import pandas as pd
import matplotlib.pyplot as plt
import json
from rich import print
import tools.ultrasound_algos as algos
import numpy as np
from rich.progress import track
import us.utils as utils
from scipy import signal
import tools.smoothing as sm
import math

import matplotlib

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

matplotlib.rc('font', size=SMALL_SIZE)          # controls default text sizes
matplotlib.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
matplotlib.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
matplotlib.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
matplotlib.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
matplotlib.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
matplotlib.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.style.use('seaborn-darkgrid')
silo_data = {
    "H1": 0.7,
    "H2": 3.618,
    "H3": 8.217,
    "Diametre": 3.64,
    "Angle (degré)": 30,
    "Offset du device": 0.335,
    "densité de moulé (kg/hl)": 75,
}


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


def wavefront_errors(data, bang_end, threshold):
    data = data[bang_end:]
    threshold = threshold * max(data)

    for i, d in enumerate(data):
        if d >= threshold:
            low = i + bang_end
            break

    for i, d in reversed(list(enumerate(data))):
        if data[i] >= threshold:
            high = i + bang_end
            break

    offset = int(abs(low - high) / 2)
    error = high - low

    return error


def scale(val, src, dst):
    """
    Scale the given value from the scale of src to the scale of dst.
    """
    return ((val - src[0]) / (src[1] - src[0])) * (dst[1] - dst[0]) + dst[0]


if __name__ == "__main__":
    data = pd.read_csv("data/agco/P2C-017.csv", converters={"rawdata": json.loads, "AcquisitionTime": pd.to_datetime})
    # data = data.sort_values(by=['AcquisitionTime'], ascending=False)
    data = data.iloc[3000:]

    inds = range(0, 6)
    batch_ids = data["BatchId"].unique()
    print(f"Number of batches: {len(batch_ids)}")
    print()

    new_weights = []
    new_dates = []
    new_distances = []
    weights = []
    dates = []
    distances = []
    computed_distances = []
    new_echos = []
    computed_tofs = []
    computed_distances_low = []
    computed_distances_high = []
    computed_vols = []
    computed_vols_lowpass = []
    computed_vols_error = []
    mean_distances = []
    j = -1
    for bid in track(batch_ids):
        j += 1
        batch_data = data[data["BatchId"] == bid]
        batch_data = batch_data.sort_values(by="AcquisitionTime")
        best_weight = batch_data.iloc[len(batch_data) - 1]["weight"]
        best_date = batch_data.iloc[len(batch_data) - 1]["AcquisitionTime"]
        best_distance = batch_data.iloc[len(batch_data) - 1]["distance"]
        weights.append(best_weight)
        dates.append(best_date)
        distances.append(best_distance)

        for i in range(len(batch_data)):
            raw_data = batch_data.iloc[i]["rawdata"]
            config_name = batch_data.iloc[i]["config"].split("p")
            pulse_count = 31
            if len(config_name) > 1:
                pulse_count == int(config_name[1])
            bang_end = algos.detect_main_bang_end(raw_data, pulse_count)
            quality = algos.auto_gain_detection(np.array(raw_data), bang_end)

            if quality == 0 or i == len(batch_data) - 1:
                new_best_weight = batch_data.iloc[i]["weight"]
                new_best_date = batch_data.iloc[i]["AcquisitionTime"]
                new_best_distance = batch_data.iloc[i]["distance"]
                raw_data = batch_data.iloc[i]["rawdata"]
                temperature = batch_data.iloc[i]["temperature"]
                new_weights.append(new_best_weight)
                new_dates.append(new_best_date)
                new_distances.append(new_best_distance)
                new_echos.append(batch_data.iloc[i]["echo"] / 2)

                cutoff_freq = 150
                b, a = signal.butter(2, cutoff_freq / 250000, 'lowpass', analog=False)
                lowpass = signal.filtfilt(b, a, raw_data)

                bang_end = algos.detect_main_bang_end(raw_data, pulse_count)
                tof_raw = algos.wavefront(raw_data, temperature, 0.5, 1.5, pulse_count)
                tof = algos.wavefront(lowpass, temperature, 0.75, 1.5, pulse_count)
                tof_low = algos.wavefront(raw_data, temperature, 0.25, 1.5, pulse_count)
                tof_high = algos.wavefront(raw_data, temperature, 0.9, 1.5, pulse_count)
                error_tof = (tof_high - tof_low) / 2

                dist_offset = utils.tof_to_dist2(tof_raw, temperature) * 2000
                dist_lowpass = utils.tof_to_dist2(tof, temperature) * 2000
                dist_low = utils.tof_to_dist2(tof - error_tof, temperature) * 2000
                dist_high = utils.tof_to_dist2(tof + error_tof, temperature) * 2000
                dist_mean = utils.tof_to_dist2((tof + tof_raw + tof_low + tof_high) / 4, temperature) * 2000
                error_dist = utils.tof_to_dist2(error_tof, temperature) * 2000

                error_vol = dist_to_volume((dist_offset + error_dist) / 1000, "allo", silo_data)
                vol = dist_to_volume(dist_offset / 1000, "allo", silo_data)
                vol_lowpass = dist_to_volume(dist_lowpass / 1000, "allo", silo_data)
                error_vol = abs(error_vol - vol)

                if j > 0:
                    prev_vol = computed_vols[-1]
                    alpha = min(1, (error_vol * 2) / 8)
                    vol = alpha * prev_vol + (1 - alpha) * vol

                computed_vols.append(vol)
                computed_vols_lowpass.append(vol_lowpass)
                computed_vols_error.append(error_vol)
                computed_distances.append(dist_offset)
                computed_distances_low.append(dist_low)
                computed_distances_high.append(dist_high)
                mean_distances.append(dist_mean)
                computed_tofs.append(tof)

                if j in []:
                    plt.plot(raw_data)
                    plt.plot(lowpass)
                    plt.axvline(tof)
                    plt.show()
                break

    plt.subplot(2, 1, 1)
    plt.title("Silo P2C-017", fontsize=20)
    plt.xlabel("Time")
    plt.ylabel("Grain weight [tons]")
    # plt.plot(dates, weights, label="current")
    # plt.plot(new_weights, label="latest")
    plt.plot(dates, sm.smooth_all(dates, weights, 40, min_fill_value=6), "--", label="latest smoothed")
    # plt.fill_between(range(len(weights)), np.array(computed_vols) - np.array(computed_vols_error),
    #                 np.array(computed_vols) + np.array(computed_vols_error), alpha=0.2, color="red")
    plt.plot(dates, computed_vols, label="computed")
    plt.plot(dates, computed_vols_lowpass, label="computed lowpass")
    # plt.plot(sm.smooth_all(dates, computed_vols, 40, min_fill_value=6), "--", label="computed smoothed")
    # plt.legend(loc="best")
    # plt.subplot(3, 1, 2)
    # plt.plot(new_echos, label="latest")
    # plt.plot(computed_echos, label="computed")
    plt.legend(loc="best")
    plt.subplot(2, 1, 2)
    plt.plot(dates, distances, label="current")
    plt.plot(new_dates, new_distances, label="latest")
    plt.plot(dates, mean_distances, "--", label="mean")
    mask = np.isfinite(computed_distances)
    plt.plot(np.array(new_dates), np.array(computed_distances), "x", color="green", label="Computed")
    plt.plot(np.array(new_dates), np.array(computed_distances), color="green", label="Computed")
    # plt.fill_between(new_dates, computed_distances_low, computed_distances_high, color="red", alpha=0.2)
    plt.legend(loc="best")
    plt.show()
    exit()

    for j, bid in enumerate(batch_ids):
        batch_data = data[data["BatchId"] == bid]
        batch_data = batch_data.sort_values(by="AcquisitionTime")

        if batch_data.iloc[0]["AcquisitionTime"].tz_localize(None) < pd.Timestamp(year=2023, month=4, day=20, hour=16, minute=0):
            print(batch_data.iloc[0]["AcquisitionTime"])
            continue
        print(f"[underline]Batch id: [green]{j}/{len(batch_ids)}[/green] - {bid}[/underline]")
        for i in range(len(batch_data)):
            plt.subplot(len(batch_data), 1, i + 1)
            print(f"[green]{str(batch_data.iloc[i]['AcquisitionTime']).split('.')[0]}[/green] - [blue]{batch_data.iloc[i]['config']}[/blue] - [red]{batch_data.iloc[i]['quality']}[/red]")
            plt.plot(batch_data.iloc[i]["rawdata"])
            plt.ylabel(f"{batch_data.iloc[i]['config']} - {batch_data.iloc[i]['quality']}")
        print()
        plt.show()
