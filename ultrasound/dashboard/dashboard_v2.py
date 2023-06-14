
import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import json
import tools.ultrasound_algos as algos
import tools.utils as utils
import tools.fill_prediction_agco as pred_agco
from scipy import signal
from enum import Enum
import plotly
import tools.smoothing as sm
from plotly.subplots import make_subplots
import glob
import os
import compute

colors = plotly.colors.DEFAULT_PLOTLY_COLORS

silo_data_path = "data/dashboard/silos/"
loadcell_data_path = "data/dashboard/loadcells/"
loadcell_col_name = "LoadCellWeight_t"

main_fig = None
main_plot_spot = None


class Silos(Enum):
    AGCO = "Mpass-10"
    A1486A = "Avinor-1486A"


def get_silo_metadata(silo):
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


def load_data(silo_name, silo_path, loadcell_path):
    silo_data = pd.read_csv(f"{silo_path}/{silo_name}.csv", converters={"AcquisitionTime": pd.to_datetime})
    loadcell_file = f"{loadcell_path}/{silo_name}.csv"
    loadcell_data = None
    if os.path.isfile(loadcell_file):
        loadcell_data = pd.read_csv(loadcell_file, converters={"AcquisitionTime": pd.to_datetime})
    return silo_data, loadcell_data


def write_data(df, path):
    df.to_csv(path, index=False)


def plot_main(data, data_lc, cols_to_plot, silo_metadata, density, silo_name, lc_perc=False, fill_prediction=False):
    global main_fig, main_plot_spot

    st.header(f"Silo: {silo_name}")
    fig = go.Figure(layout_legend_groupclick="toggleitem")
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
        output_name = f"data/dashboard/output/{silo_name}_{col}.csv"
        if not os.path.isfile(output_name):
            resampled.to_csv(output_name)
        # resampled = resampled[:-9]
        fig.add_scatter(x=resampled.index, y=resampled[col], name=f"Resampled {col}",
                        mode="lines+markers", line_dash="dash", line_color=color, legendgroup=col)

        if fill_prediction:
            pred_time, pred_value = pred_agco.predict_next_fill_agco(
                resampled.index.values,
                resampled[col].values,
                tons_fill_threshold=0)
            vline(fig, pred_time, color="green", name="Empty prediction", min=-3)
            fig.add_scatter(x=[resampled.index[-1], pred_time - pd.Timedelta(days=0.6)],
                            y=[resampled[col][-1], pred_value], mode='lines', line_color="rgba(255, 0, 0, 0.0)", opacity=0.1, showlegend=False)
            fig.add_scatter(x=[resampled.index[-1], pred_time + pd.Timedelta(days=0.8)], y=[resampled[col][-1], pred_value], mode="lines",
                            line_color="rgba(255, 0, 0, 0.0)", opacity=0.1, fill='tonexty', fillcolor="rgba(255, 0, 0, 0.4)", name="Uncertainty")
            fig.add_scatter(x=[resampled.index[-1], pred_time], y=[resampled[col][-1], pred_value], mode="lines",
                            line_dash="dash", line_color="green", opacity=0.8, showlegend=False)

            fig.add_scatter(x=[pred_time], y=[0], showlegend=False)

        smoothed = sm.smooth_all(
            resampled.index.values, resampled[col].values, 50, exp_filter_timestep=4, exp_filter_tau=2, spike_filter_max_perc=5,
            spike_filter_num_change=2, min_fill_value=4)

        fig.add_scatter(x=resampled.index, y=smoothed, name=f"Smoothed Resampled {col}",
                        mode="lines+markers", line_dash="dash", line_color=color, legendgroup=col)

    if lc_perc:
        data_lc[loadcell_col_name] = data_lc[loadcell_col_name] / (silo_metadata["BinVolume"] / 100 * density) * 100
    if max(data_lc[loadcell_col_name]) > ymax:
        ymax = max(data_lc[loadcell_col_name])
    fig.add_scatter(x=data_lc["AcquisitionTime"], y=data_lc[loadcell_col_name], name="Loadcell", mode="markers", marker_size=5, marker_color="red")
    fig.update_yaxes(range=[-3, ymax + 5])

    main_fig = fig
    main_plot_spot = st.empty()


def main():
    available_silos = [name.split("/")[-1].split(".")[0] for name in glob.glob("data/dashboard/silos/*.csv")]
    st.set_page_config(layout="wide")
    st.write("# Silo viz tool")
    cols = st.columns(3)
    selected_silos = cols[0].multiselect("Which silo to visualize", available_silos)
    density = cols[2].number_input("Density", min_value=60, max_value=80, value=70)
    silo_type = cols[1].multiselect("Silo data to use", [Silos.AGCO, Silos.A1486A], max_selections=1, default=Silos.AGCO)
    if len(silo_type) > 0:
        silo_metadata = get_silo_metadata(silo_type[0])

    if len(selected_silos) == 0:
        return

    for silo_name in selected_silos:
        st.divider()
        st.title(silo_name)
        silo_data, lc_data = load_data(silo_name, silo_data_path, loadcell_data_path)
        cols = st.columns([5, 1])
        cols_to_plot = cols[0].multiselect("Sensor columns to show", silo_data.columns, key=f"select{silo_name}")
        lc_perc = cols[1].checkbox("Loadcell %", key=f"check{silo_name}")
        compute_algos = cols[1].checkbox("Compute algos", key=f"check2{silo_name}")
        fill_prediction = cols[1].checkbox("Compute fill prediction", key=f"check3{silo_name}")
        if compute_algos:
            my_bar = st.progress(0, text="Computing algos...")
            write_to_file = False
            if "wf_weight" not in silo_data.columns:
                my_bar.progress(10, text="Computing wf...")
                wf_index, wf_weight, wf_perc = compute.compute_wavefront(silo_data, density, silo_metadata)
                silo_data["wf_weight"] = wf_weight
                silo_data["wf_index"] = wf_index
                silo_data["wf_perc"] = wf_perc
                write_to_file = True
            if "lowpass_weight" not in silo_data.columns:
                my_bar.progress(20, text="Computing lowpass...")
                lowpass_index, lowpass_weight, lowpass_perc = compute.compute_lowpass(silo_data, density, silo_metadata)
                silo_data["lowpass_weight"] = lowpass_weight
                silo_data["lowpass_index"] = lowpass_index
                silo_data["lowpass_perc"] = lowpass_perc
                write_to_file = True
            if "cdm_weight" not in silo_data.columns:
                my_bar.progress(30, text="Computing cdm...")
                cdm_index, cdm_weight, cdm_perc = compute.compute_cdm(silo_data, density, silo_metadata)
                silo_data["cdm_weight"] = cdm_weight
                silo_data["cdm_index"] = cdm_index
                silo_data["cdm_perc"] = cdm_perc
                write_to_file = True
            if "best_weight" not in silo_data.columns:
                my_bar.progress(40, text="Computing best...")
                best_index, best_weight, best_perc = compute.compute_best(silo_data, density, silo_metadata)
                silo_data["best_weight"] = best_weight
                silo_data["best_index"] = best_index
                silo_data["best_perc"] = best_perc
                write_to_file = True

            if write_to_file:
                my_bar.progress(50, text="writing to file...")
                write_data(silo_data, f"{silo_data_path}/{silo_name}.csv")
            my_bar.progress(100, text="Done.")

        if len(cols_to_plot) == 0:
            continue
        plot_main(silo_data, lc_data, cols_to_plot, silo_metadata, density, silo_name, lc_perc, fill_prediction)
        st.plotly_chart(main_fig, use_container_width=True)


main()
