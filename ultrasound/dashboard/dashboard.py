import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import json
import tools.ultrasound_algos as algos
import tools.utils as utils
from scipy import signal
from enum import Enum
import plotly
import tools.smoothing as sm
from plotly.subplots import make_subplots

selected_raw_datas = []
raw_data_col = []
temperature_col = []
pulse_count_col = []
density = None
main_fig = None
main_plot_spot = None
silo_type = []
silo_data = None
lc_silo_name = None

colors = plotly.colors.DEFAULT_PLOTLY_COLORS


class Silos(Enum):
    AGCO = "Mpass-10"
    A1486A = "Avinor-1486A"


def get_silo_data(silo):
    silo_datas = pd.read_csv("data/dashboard/LocationMetadata.csv")
    if silo.value == Silos.AGCO.value:
        silo_data = silo_datas.loc[silo_datas["LocationName"] == "MPass-10"].squeeze()
        return silo_data.to_dict()
    elif silo.value == Silos.A1486A.value:
        silo_data = silo_datas[silo_datas["LocationName"] == "Avinor-1486A"].squeeze()
        return silo_data.to_dict()
    return


def vline(fig, x, color="None", name=None, showlegend=True, min=0, max=255):
    fig.add_scatter(x=[x, x], y=[min, max], name=name, line_color=color, mode="lines", line_dash="dash", showlegend=showlegend)


def read_data():
    file = st.file_uploader("Open silo data", type="csv")
    file_lc = st.file_uploader("Open loadcell data", type="csv")
    return file, file_lc


@st.cache_data
def compute_wavefront(data):
    global temperature_col, pulse_count_col, raw_data_col, silo_data
    wf_index = data.apply(lambda row: algos.wavefront(json.loads(row[raw_data_col[0]]),
                                                      row[temperature_col[0]], 0.5, 10, row[pulse_count_col[0]]) * 2, axis=1)
    data["wf_index"] = wf_index
    wf_dist = data.apply(lambda row: utils.tof_to_dist(row["wf_index"], row[temperature_col[0]]), axis=1)
    data["wf_dist"] = wf_dist
    wf_weight = data.apply(lambda row: utils.dist_to_volume_agco(row["wf_dist"], silo_data) * density / 100, axis=1)
    wf_perc = data.apply(lambda row: utils.dist_to_volume_agco(row["wf_dist"], silo_data) * 100 / silo_data["BinVolume"], axis=1)
    return wf_index, wf_weight, wf_perc


@st.cache_data
def compute_lowpass(data):
    global temperature_col, pulse_count_col, raw_data_col, silo_data
    cutoff_freq = 150
    b, a = signal.butter(2, cutoff_freq / 250000, 'lowpass', analog=False)
    lowpass_index = data.apply(lambda row: algos.wavefront(signal.filtfilt(
        b, a, json.loads(row[raw_data_col[0]])), row[temperature_col[0]], 0.75, 10, row[pulse_count_col[0]]) * 2, axis=1)
    data["lowpass_index"] = lowpass_index
    lowpass_dist = data.apply(lambda row: utils.tof_to_dist(row["lowpass_index"], row[temperature_col[0]]), axis=1)
    data["lowpass_dist"] = lowpass_dist
    lowpass_weight = data.apply(lambda row: utils.dist_to_volume_agco(row["lowpass_dist"], silo_data) * density / 100, axis=1)
    lowpass_perc = data.apply(lambda row: utils.dist_to_volume_agco(row["lowpass_dist"], silo_data) * 100 / silo_data["BinVolume"], axis=1)
    return lowpass_index, lowpass_weight, lowpass_perc


@st.cache_data
def compute_cdm(data):
    global temperature_col, pulse_count_col, raw_data_col, silo_data
    cdm_index = data.apply(lambda row: algos.center_of_mass(json.loads(row[raw_data_col[0]]), row[pulse_count_col[0]]) * 2, axis=1)
    data["cdm_index"] = cdm_index
    cdm_dist = data.apply(lambda row: utils.tof_to_dist(row["cdm_index"], row[temperature_col[0]]), axis=1)
    data["cdm_dist"] = cdm_dist
    cdm_weight = data.apply(lambda row: utils.dist_to_volume_agco(row["cdm_dist"], silo_data) * density / 100, axis=1)
    cdm_perc = data.apply(lambda row: utils.dist_to_volume_agco(row["cdm_dist"], silo_data) / silo_data["BinVolume"] * 100, axis=1)
    return cdm_index, cdm_weight, cdm_perc


@st.cache_data
def compute_cdm_lowpass(data):
    global temperature_col, pulse_count_col, raw_data_col, silo_data
    cutoff_freq = 150
    b, a = signal.butter(2, cutoff_freq / 250000, 'lowpass', analog=False)
    cdm_lowpass_index = data.apply(lambda row: algos.center_of_mass(signal.filtfilt(
        b, a, json.loads(row[raw_data_col[0]])), row[pulse_count_col[0]]) * 2, axis=1)
    data["cdm_lowpass_index"] = cdm_lowpass_index
    cdm_lowpass_dist = data.apply(lambda row: utils.tof_to_dist(row["cdm_lowpass_index"], row[temperature_col[0]]), axis=1)
    data["cdm_lowpass_dist"] = cdm_lowpass_dist
    cdm_lowpass_weight = data.apply(lambda row: utils.dist_to_volume_agco(row["cdm_lowpass_dist"], silo_data) * density / 100, axis=1)
    cdm_lowpass_perc = data.apply(lambda row: utils.dist_to_volume_agco(row["cdm_lowpass_dist"], silo_data) / silo_data["BinVolume"] * 100, axis=1)
    return cdm_lowpass_index, cdm_lowpass_weight, cdm_lowpass_perc


@ st.cache_data
def compute_best(data):
    global temperature_col, pulse_count_col, raw_data_col, silo_data
    best_dist = []
    best_weight = []
    best_perc = []
    cutoff_freq = 150
    b, a = signal.butter(2, cutoff_freq / 250000, 'lowpass', analog=False)
    for i in range(len(data)):
        row = data.iloc[i]
        pulse = row[pulse_count_col[0]]
        temperature = row[temperature_col[0]]
        raw_data = json.loads(row[raw_data_col[0]])

        lowpass = signal.filtfilt(b, a, raw_data)
        lowpass_bang_end = algos.detect_main_bang_end(lowpass, pulse)
        if max(raw_data[lowpass_bang_end:]) < 20:
            best_dist.append(utils.tof_to_dist(row["WF_index"], temperature))
            best_weight.append(row["WF_weight"])
            best_perc.append(row["WF_perc"])
        else:
            best_dist.append(utils.tof_to_dist(row["lowpass_index"], temperature))
            best_weight.append(row["lowpass_weight"])
            best_perc.append(row["lowpass_perc"])
    return best_dist, best_weight, best_perc


@ st.cache_data
def compute_enveloppe(data):
    global temperature_col, pulse_count_col, raw_data_col, silo_data
    env_index = data.apply(
        lambda row: algos.wavefront(
            algos.enveloppe(json.loads(row[raw_data_col[0]]),
                            row[pulse_count_col[0]]),
            row[temperature_col[0]],
            0.75, 10, row[pulse_count_col[0]]) * 2, axis=1)
    data["env_index"] = env_index
    env_dist = data.apply(lambda row: utils.tof_to_dist(row["env_index"], row[temperature_col[0]]), axis=1)
    data["env_dist"] = env_dist
    env_weight = data.apply(lambda row: utils.dist_to_volume_agco(row["env_dist"], silo_data) * density / 100, axis=1)
    return env_index, env_weight


def data_selections(data, data_lc):
    global temperature_col, pulse_count_col, raw_data_col, density, silo_data, silo_type
    cols = st.columns(5)

    density = cols[0].number_input("Density", min_value=60, max_value=80, value=70)
    temperature_col = cols[1].multiselect("Temperature column", data.columns, max_selections=1, default="temperature")
    pulse_count_col = cols[2].multiselect("Pulse count column", data.columns, max_selections=1, default="pulseCount")
    raw_data_col = cols[3].multiselect("Raw data column", data.columns, max_selections=1, default="rawdata")
    silo_type = cols[4].multiselect("Silo data to use", [Silos.AGCO, Silos.A1486A], max_selections=1, default=Silos.AGCO)
    if len(silo_type) > 0:
        silo_data = get_silo_data(silo_type[0])


def plot_main(data, data_lc):
    global raw_data_col, selected_raw_datas, main_fig, main_plot_spot, lc_silo_name
    col1, col2, col3, col4 = st.columns(4)
    cols_to_plot = col1.multiselect("Sensor columns to show", data.columns)
    lc_col_to_plot = col2.multiselect("Loadcell column to show", data_lc.columns, max_selections=1)
    lc_perc = col4.checkbox("Loadcell %")

    if "LocationName" in data_lc.columns:
        lc_silo_name = col3.multiselect("Silo name", data_lc["LocationName"].unique(), max_selections=1)
    if len(cols_to_plot) == 0:
        return False

    fig = go.Figure(layout_title="Weights", layout_legend_groupclick="toggleitem")
    ymax = 0
    for i, col in enumerate(cols_to_plot):
        color = colors[i % len(colors)]
        spike_filtered = sm.generic_iir_filter(data[col].values, sm.spike_filter, {
            "maximum_change_perc": 10, "number_of_changes": 3, "count": 0, "bin_max": max(data[col])})
        data[col] = spike_filtered
        if max(data[col]) > ymax:
            ymax = max(data[col])

        fig.add_scatter(x=data["AcquisitionTime"], y=data[col], name=col, line_color=color, legendgroup=col, legendgrouptitle_text=col)

        resampled = data.set_index("AcquisitionTime").resample("4H").mean()
        fig.add_scatter(x=resampled.index, y=resampled[col], name=f"Resampled {col}",
                        mode="lines+markers", line_dash="dash", line_color=color, legendgroup=col)
        # smoothed = sm.smooth_all(data["AcquisitionTime"], data[col], silo_data["BinVolume"], exp_filter_timestep=4)
        # data[col] = smoothed
        # smoothed_resampled = data.set_index("AcquisitionTime").resample("4H").mean()

        # fig.add_scatter(x=smoothed_resampled.index, y=smoothed_resampled[col], name=f"Smoothed Resampled {col}",
        #                mode="lines+markers", line_dash="dash", line_color=color, legendgroup=col)
    fig.update_yaxes(range=[-5, ymax + 5])

    if len(lc_col_to_plot) > 0:
        if lc_perc:
            data_lc[lc_col_to_plot[0]] = data_lc[lc_col_to_plot[0]] / (silo_data["BinVolume"] / 100 * density) * 100
        if "LocationName" in data_lc.columns and len(lc_silo_name) > 0:
            data_lc = data_lc[data_lc["LocationName"] == lc_silo_name[0]]
            fig.add_scatter(x=data_lc["AcquisitionTime"], y=data_lc[lc_col_to_plot[0]], name="Loadcell", mode="markers")
        elif "LocationName" not in data_lc.columns:
            fig.add_scatter(x=data_lc["AcquisitionTime"], y=data_lc[lc_col_to_plot[0]], name="Loadcell", mode="markers")

    main_fig = fig
    main_plot_spot = st.empty()
    # plot_error(data, data_lc)
    return True


def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))


def plot_raw(data, data_lc):
    global selected_raw_datas, main_fig, raw_data_col, temperature_col, pulse_count_col, lc_silo_name, density, silo_data
    selected_raw_datas = st.multiselect("Raw data to plot", reversed(data["AcquisitionTime"]))
    for s in selected_raw_datas:
        fig = go.Figure(layout_title=f"Raw data - {s}")
        row = data.loc[data["AcquisitionTime"] == s].reset_index().iloc[0]
        raw_data = json.loads(row.at["rawdata"])
        pulse = row[pulse_count_col[0]]
        temperature = row[temperature_col[0]]
        bang_end = algos.detect_main_bang_end(raw_data, pulse)
        wf = algos.wavefront(raw_data, temperature, 0.5, 10, pulse)
        cdm = algos.center_of_mass(raw_data, pulse)
        fig.add_scatter(y=raw_data, name="Raw data")

        cutoff_freq = 150
        b, a = signal.butter(2, cutoff_freq / 250000, 'lowpass', analog=False)
        lowpass = signal.filtfilt(b, a, raw_data)
        bang_end_lowpass = algos.detect_main_bang_end(lowpass, pulse)
        fig.add_scatter(y=lowpass, name="Lowpass", line_color="red")
        vline(fig, algos.wavefront(lowpass, row[temperature_col[0]],
              0.75, 10, row[pulse_count_col[0]]), color="red", name="Lowpass wavefront")

        cdm_lowpass = algos.center_of_mass(lowpass, pulse)

        enveloppe = algos.enveloppe(raw_data, pulse)
        fig.add_scatter(y=enveloppe, name="Enveloppe", line_color="orange")

        vline(fig, wf, color="pink", name="Wavefront")
        vline(fig, cdm, color="purple", name="Center of mass")
        vline(fig, cdm_lowpass, color="lightblue", name="Center of mass")
        vline(fig, bang_end, color="gray", name="Bang end")
        vline(fig, bang_end_lowpass, color="orange", name="Bang end lowpass")

        # data_lc = data_lc[data_lc["LocationName"] == lc_silo_name[0]]
        closest_lc = nearest(data_lc["AcquisitionTime"], s)
        lc_weight = data_lc[data_lc["AcquisitionTime"] == closest_lc]["LoadCellWeight_t"].item()
        lc_tof = utils.weight_to_tof(lc_weight, silo_data, density, temperature) / 2
        vline(fig, lc_tof, color="green", name="Loadcell")

        st.plotly_chart(fig, use_container_width=True)

        vline(main_fig, s, color="gray", name=None, min=0, max=40, showlegend=False)


def plot_error(data, data_lc):
    errors = []
    for i in range(len(data)):
        row = data.iloc[i]
        t = row["AcquisitionTime"]
        weight = row["best_weight"]

        closest_lc = nearest(data_lc["AcquisitionTime"], t)
        lc_weight = data_lc[data_lc["AcquisitionTime"] == closest_lc]["LoadCellWeight_t"].item()

        errors.append(abs(weight - lc_weight))
    data["best_error"] = errors

    col1, col2 = st.columns(2)
    fig = go.Figure(layout_title="Errors", layout_xaxis_title="Abs error", layout_yaxis_title="Count")
    fig.add_trace(go.Histogram(histfunc="avg", x=data["best_error"], name="Abs error"))
    col1.plotly_chart(fig, use_container_width=True)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
# go.Figure(layout_title="Errors", layout_xaxis_title="Distance", layout_yaxis_title="Mean abs error")
    fig.add_trace(go.Histogram(histfunc="avg", x=data["best_dist"], y=data["best_error"], name="Abs error mean"))
    fig.add_trace(go.Histogram(histfunc="count", x=data["best_dist"], y=data["best_error"], name="Abs error count"), secondary_y=True)
    col2.plotly_chart(fig, use_container_width=True)


def main():
    global temperature_col, pulse_count_col, raw_data_col, main_fig, main_plot_spot, silo_data, silo_type
    st.set_page_config(layout="wide")
    st.write("# Silo viz tool")
    show_df = st.checkbox("Show dataframes")
    file, file_lc = read_data()
    if file is not None and file_lc is not None:
        data = pd.read_csv(file, converters={"AcquisitionTime": pd.to_datetime})
        data_lc = pd.read_csv(file_lc, converters={"AcquisitionTime": pd.to_datetime})
        data_selections(data, data_lc)

        if len(temperature_col) > 0 and len(pulse_count_col) and len(raw_data_col) > 0 and len(silo_type) > 0:
            col1, col2, col3, col4 = st.columns(4)
            comp_wf = col1.checkbox("Compute wavefront")
            comp_lowpass = col2.checkbox("Compute lowpass")
            comp_cdm = col3.checkbox("Compute CDM")
            comp_env = col4.checkbox("Compute enveloppe")
            if comp_wf:
                wf_index, wf_weight, wf_perc = compute_wavefront(data)
                data["WF_weight"] = wf_weight
                data["WF_index"] = wf_index
                data["WF_perc"] = wf_perc
            if comp_lowpass:
                lowpass_index, lowpass_weight, lowpass_perc = compute_lowpass(data)
                data["lowpass_weight"] = lowpass_weight
                data["lowpass_index"] = lowpass_index
                data["lowpass_perc"] = lowpass_perc
            if comp_cdm:
                cdm_index, cdm_weight, cdm_perc = compute_cdm(data)
                data["cdm_weight"] = cdm_weight
                data["cdm_index"] = cdm_index
                data["cdm_perc"] = cdm_perc
                cdm_lowpass_index, cdm_lowpass_weight, cdm_lowpass_perc = compute_cdm_lowpass(data)
                data["cdm_lowpass_weight"] = cdm_lowpass_weight
                data["cdm_lowpass_index"] = cdm_lowpass_index
                data["cdm_lowpass_perc"] = cdm_lowpass_perc
            if comp_env:
                env_index, env_weight = compute_enveloppe(data)
                data["env_weight"] = env_weight
                data["env_index"] = env_index
            if comp_wf and comp_lowpass:
                best_dist, best_weight, best_perc = compute_best(data)
                data["best_dist"] = best_dist
                data["best_weight"] = best_weight
                data["best_perc"] = best_perc

        if plot_main(data, data_lc):
            plot_raw(data, data_lc)
            with main_plot_spot:
                st.plotly_chart(main_fig, use_container_width=True)

        if show_df:
            st.write(data)
            st.write(data_lc)
    else:
        st.write("Please select a file for the data and loadcell data")


main()
