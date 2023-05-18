import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import json
import tools.ultrasound_algos as algos
import tools.utils as utils
from scipy import signal
from enum import Enum, auto

selected_raw_datas = []
raw_data_col = []
temperature_col = []
pulse_count_col = []
density = None
main_fig = None
main_plot_spot = None
silo_type = []
silo_data = None


class Silos(Enum):
    AGCO = "Mpass-10"
    A1486A = "Avinor-1486A"


def get_silo_data(silo):
    silo_datas = pd.read_csv("data/dashboard/LocationMetadata.csv")
    if silo.value == Silos.AGCO.value:
        st.write(f"1{silo}")
        silo_data = silo_datas[silo_datas["LocationName"] == "Mpass-10"].squeeze()
        st.write(silo_data)
        st.write(type(silo_data))
        return silo_data.to_dict()
    elif silo == Silos.A1486A:
        silo_data = silo_datas[silo_datas["LocationName"] == "Avinor-1486A"].reset_index().iloc[0]
        return utils.agco_silo_data
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
    print(silo_data)
    wf_index = data.apply(lambda row: algos.wavefront(json.loads(row[raw_data_col[0]]),
                                                      row[temperature_col[0]], 0.5, 10, row[pulse_count_col[0]]) * 2, axis=1)
    data["wf_index"] = wf_index
    wf_dist = data.apply(lambda row: utils.tof_to_dist(row["wf_index"], row[temperature_col[0]]), axis=1)
    data["wf_dist"] = wf_dist
    wf_weight = data.apply(lambda row: utils.dist_to_volume_agco(row["wf_dist"], silo_data) * density / 100, axis=1)
    return wf_index, wf_weight


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
    return lowpass_index, lowpass_weight


@st.cache_data
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
    temperature_col = cols[1].multiselect("Temperature column", data.columns, max_selections=1)
    pulse_count_col = cols[2].multiselect("Pulse count column", data.columns, max_selections=1)
    raw_data_col = cols[3].multiselect("Raw data column", data.columns, max_selections=1)
    silo_type = cols[4].multiselect("Silo data to use", [Silos.AGCO, Silos.A1486A], max_selections=1)
    if len(silo_type) > 0:
        silo_data = get_silo_data(silo_type[0])
        st.write(silo_data)


def plot_main(data, data_lc):
    global raw_data_col, selected_raw_datas, main_fig, main_plot_spot
    col1, col2, col3 = st.columns(3)
    cols_to_plot = col1.multiselect("Sensor columns to show", data.columns)
    lc_col_to_plot = col2.multiselect("Loadcell column to show", data_lc.columns, max_selections=1)

    if "LocationName" in data_lc.columns:
        silo_name = col3.multiselect("Silo name", data_lc["LocationName"].unique(), max_selections=1)
    if len(cols_to_plot) == 0:
        return False

    fig = go.Figure(layout_title="Weights")
    for col in cols_to_plot:
        fig.add_scatter(x=data["AcquisitionTime"], y=data[col], name=col)

    if len(lc_col_to_plot) > 0:
        if "LocationName" in data_lc.columns and len(silo_name) > 0:
            data_lc = data_lc[data_lc["LocationName"] == silo_name[0]]
            fig.add_scatter(x=data_lc["AcquisitionTime"], y=data_lc[lc_col_to_plot[0]], name="Loadcell")
        elif "LocationName" not in data_lc.columns:
            fig.add_scatter(x=data_lc["AcquisitionTime"], y=data_lc[lc_col_to_plot[0]], name="Loadcell")

    main_fig = fig
    main_plot_spot = st.empty()
    return True


def plot_raw(data):
    global selected_raw_datas, main_fig, raw_data_col, temperature_col, pulse_count_col
    selected_raw_datas = st.multiselect("Raw data to plot", reversed(data["AcquisitionTime"]))
    for s in selected_raw_datas:
        fig = go.Figure(layout_title=f"Raw data - {s}")
        row = data.loc[data["AcquisitionTime"] == s].reset_index().iloc[0]
        raw_data = json.loads(row.at["rawdata"])
        pulse = row["pulseCount"]
        temperature = row["temperature"]
        bang_end = algos.detect_main_bang_end(raw_data, pulse)
        wf = algos.wavefront(raw_data, temperature, 0.5, 10, pulse)
        fig.add_scatter(y=raw_data, name="Raw data")

        cutoff_freq = 150
        b, a = signal.butter(2, cutoff_freq / 250000, 'lowpass', analog=False)
        lowpass = signal.filtfilt(b, a, raw_data)
        fig.add_scatter(y=lowpass, name="Lowpass", line_color="red")
        vline(fig, algos.wavefront(lowpass, row[temperature_col[0]],
              0.75, 10, row[pulse_count_col[0]]), color="red", name="Lowpass wavefront")

        enveloppe = algos.enveloppe(raw_data, pulse)
        fig.add_scatter(y=enveloppe, name="Enveloppe", line_color="orange")

        vline(fig, wf, color="pink", name="Wavefront")
        vline(fig, bang_end, color="gray", name="Bang end")
        st.plotly_chart(fig, use_container_width=True)

        vline(main_fig, s, color="gray", name=None, min=0, max=40, showlegend=False)


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
            col1, col2, col3 = st.columns(3)
            if col1.checkbox("Compute wavefront"):
                wf_index, wf_weight = compute_wavefront(data)
                data["WF_weight"] = wf_weight
                data["WF_index"] = wf_index
            if col2.checkbox("Compute lowpass"):
                lowpass_index, lowpass_weight = compute_lowpass(data)
                data["lowpass_weight"] = lowpass_weight
                data["lowpass_index"] = lowpass_index
            if col3.checkbox("Compute enveloppe"):
                env_index, env_weight = compute_enveloppe(data)
                data["env_weight"] = env_weight
                data["env_index"] = env_index

        if plot_main(data, data_lc):
            plot_raw(data)
            with main_plot_spot:
                st.plotly_chart(main_fig, use_container_width=True)

        if show_df:
            st.write(data)
            st.write(data_lc)
    else:
        st.write("Please select a file for the data and loadcell data")


main()
