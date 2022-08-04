from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import signal

import us.algos as algos


def plot_raw_ultrasound(df: pd.DataFrame, manual_data: pd.DataFrame, show: bool = True):
    values = df["ultrasons_data"]
    plt.plot(values)

    print(values.values.shape)
    down = 10
    downsampled_raw = signal.resample_poly(values.values.astype(float), 1, down)
    print(downsampled_raw.shape)
    plt.plot(downsampled_raw, "-", label="downsampled_raw")
    plt.axvline(manual_data["TOF_ManuealReading"] / down, linestyle="dashed", color="green", label="Manual TOF")
    plt.axvline(manual_data["TOF_ManuealReading"], linestyle="dashed", color="green", label="Manual TOF")
    plt.show()

    plt.plot(downsampled_raw, "-", label="downsampled_raw")
    plt.axvline(manual_data["TOF_ManuealReading"], linestyle="dashed", color="green", label="Manual TOF")
    cm_index = manual_data["measured_distance_in_mm"] * 2* 1000000 / 1000 / manual_data["sound_speed"]
    wf_index = manual_data["wavefront_distance_in_mm"] *2*  1000000 / 1000 / manual_data["sound_speed"]
    plt.axvline(cm_index, linestyle="dashed", color="yellow", label="CM TOF")
    plt.axvline(wf_index, linestyle="dashed", color="orange", label="WF TOF")

    print(values)
    noise_threshold = algos.NoiseThresholdv1().process(values)
    result_mainbang = algos.MainBangDetectorv1().process(values, noise_threshold)

    result_CM_lin = algos.CenterOfMassLin().process(values[result_mainbang["main_bang_end"]:], noise_threshold)
    result_CM_quad = algos.CenterOfMassQuad().process(values[result_mainbang["main_bang_end"]:], noise_threshold)

    plt.axvline(result_mainbang["main_bang_start"], color="black", label="Main bang start")
    plt.axvline(result_mainbang["main_bang_end"], color="black", label="Main bang end")
    plt.axvline(result_CM_lin, color="red", linestyle="dashed", label="CM TOF Lin Computed")
    plt.axvline(result_CM_quad, color="red", linestyle="dashed", label="CM TOF Quad Computed")

    print(f"File name : {manual_data['filename']}")
    print(noise_threshold)
    print(result_mainbang)
    print(result_CM_lin)
    print(result_CM_quad)


    plt.title(f"File: {manual_data['filename']}")

    plt.legend(loc="best")
    if show:
        plt.show()


def plot_full_excel(df: pd.DataFrame):
    df = df[:2460]
    df = pd.DataFrame(df.sort_values(by="time"))
    plt.plot(df["time"], df["CM_vol"], label="CM Volume")
    plt.plot(df["time"], df["WF_vol"], label="WF Volume")
    plt.plot(df["time"], df["ManualReadingfilled"], label="Manual Volume")
    
    plt.legend(loc="best")
    plt.show()


def plot_compare_raw(raws: List[pd.DataFrame], excel_lines: List[pd.DataFrame]):
    for i, (raw, excel_line) in enumerate(zip(raws, excel_lines)):
        plt.subplot(len(raws), 1, i+1)
        plot_raw_ultrasound(raw, excel_line, show=False)
    plt.show()


def plot_dashboard_data(volumes: pd.DataFrame, temperatures: pd.DataFrame, temperatures_extern: pd.DataFrame):

    for col in volumes.columns[1:]:
        volumes[col] = volumes[col].str.replace(" t", "").replace("-∞", "0.0").astype(float)

    temperatures["temperature"] = temperatures["temperature"].str.replace(" °C", "").astype(float)

    temperatures_extern["Date/Heure (HNL)"] = pd.to_datetime(temperatures_extern["Date/Heure (HNL)"])
    temperatures_extern["Temp (°C)"] = temperatures_extern["Temp (°C)"].str.replace(",", ".").astype(float)

    plt.subplot(2, 1, 1)
    plt.plot(volumes["AcquisitionTime"].values, volumes["FeedRemaining_T"].values, label="FeedRemaining")
    plt.plot(volumes["AcquisitionTime"].values, volumes["rawFeedWeight_T"].values, label="rawFeedWeight")
    plt.xlim(volumes["AcquisitionTime"].values[0], volumes["AcquisitionTime"].values[-1])
    plt.legend(loc="best")

    plt.subplot(2, 1, 2)
    plt.plot(temperatures["AcquisitionTime"], temperatures["temperature"], label="Temperature")
    plt.plot(temperatures_extern["Date/Heure (HNL)"], temperatures_extern["Temp (°C)"], label="Temperature Extern")
    plt.xlim(volumes["AcquisitionTime"].values[0], volumes["AcquisitionTime"].values[-1])
    plt.legend(loc="best")

    plt.show()
