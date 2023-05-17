import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import json
import tools.ultrasound_algos as algos

selected_raw_datas = []


def vline(fig, x, color="None", name=None):
    fig.add_scatter(x=[x, x], y=[0, 255], name=name, line_color=color, mode="lines", line_dash="dash")


def read_data():
    file = st.file_uploader("Open silo data", type="csv")
    file_lc = st.file_uploader("Open loadcell data", type="csv")
    return file, file_lc


def click_callback(trace, points, selector):
    print(trace)
    print(points)
    print(selector)
    st.write("CLICKED")


def plot_main(data, data_lc):
    fig = go.Figure(layout_title="Weights")
    fig.add_scatter(x=data_lc["AcquisitionTime"], y=data_lc["w_t"], name="Loadcell")
    fig.add_scatter(x=data["AcquisitionTime"], y=data["PGA_weight"], name="PGA")
    fig.add_scatter(x=data["AcquisitionTime"], y=data["CDM_weight"], name="CDM")
    fig.add_scatter(x=data["AcquisitionTime"], y=data["WF_weight"], name="WF")
    # fig.data[0].on_click(click_callback)

    st.plotly_chart(fig, use_container_width=True)


def plot_raw(data):
    selected = st.multiselect("Raw data to plot", data["AcquisitionTime"])
    for s in selected:
        fig = go.Figure(layout_title=f"Raw data - {s}")
        row = data.loc[data["AcquisitionTime"] == s].reset_index().iloc[0]
        raw_data = json.loads(row.at["rawdata"])
        pulse = row["pulseCount"]
        temperature = row["temperature"]
        bang_end = algos.detect_main_bang_end(raw_data, pulse)
        wf = algos.wavefront(raw_data, temperature, 0.5, 10, pulse)
        fig.add_scatter(y=raw_data, name="Raw data")
        vline(fig, wf, color="pink", name="Wavefront")
        vline(fig, bang_end, color="red", name="Bang end")
        st.plotly_chart(fig, use_container_width=True)


def main():
    st.write("# Silo viz tool")
    show_df = st.checkbox("Show dataframes")
    file, file_lc = read_data()
    if file is not None and file_lc is not None:
        data = pd.read_csv(file, converters={"AcquisitionTime": pd.to_datetime})
        data_lc = pd.read_csv(file_lc, converters={"AcquisitionTime": pd.to_datetime})
        plot_main(data, data_lc)
        plot_raw(data)
        if show_df:
            st.write(data)
            st.write(data_lc)
    else:
        st.write("Please select a file for the data and loadcell data")


main()
