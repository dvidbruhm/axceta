import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import numpy as np

from config.config import logger
from snr.filters import realtime_savgol, realtime_regression, nonrealtime_savgol
from snr.utils import batch_dist_to_vol


def plot_data_streamlit(data: pd.DataFrame, conversion_data: pd.DataFrame) -> None:
    logger.info("Starting streamlit plot...")
    st.set_page_config(layout="wide")
    st.title('Filters experimentation on silo data')
    plotly_theme = "plotly_dark"

    if st.checkbox("Show raw data", value=False):
        st.subheader("Raw silo data")
        st.write(data)

    locations = sorted(data["LocationName"].unique())

    _, col2, _ = st.columns(3)
    with col2:
        silo_name = st.selectbox("Silo list", options=locations)
    st.subheader(f"Silo name : {silo_name}")
    _, col2, _ = st.columns(3)
    with col2:
        savgol_window = st.slider("Savgol filter window : ", 5, 51, 21, 2)
        savgol_offset = st.slider("Savgol offset : ", 1, savgol_window, 1)
        reg_nb_points = st.slider("Regression nb points : ", 5, 50, 20, 1)

    silo_data = batch_dist_to_vol(data, conversion_data, silo_name)

    resampled_silo_data = silo_data[["AcquisitionTime", "perc_filled"]].resample('1H', on="AcquisitionTime").mean()
    resampled_silo_data["AcquisitionTime"] = resampled_silo_data.index
    resampled_silo_data["AcquisitionTime_int"] = resampled_silo_data.index
    resampled_silo_data["perc_filled"] = resampled_silo_data["perc_filled"].interpolate(method="nearest")
    x = resampled_silo_data["AcquisitionTime_int"].view('int64').values // 10 ** 9
    y = resampled_silo_data["perc_filled"].values
    resampled_silo_data["filtered_perc1"] = realtime_regression(x, y, reg_nb_points)
    resampled_silo_data["filtered_perc2"] = realtime_savgol(x, y, savgol_window, 2, 0, savgol_offset)
    resampled_silo_data["filtered_perc3"] = nonrealtime_savgol(x, y, savgol_window, 2, 0)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig.update_layout(template=plotly_theme)
    fig.update_layout(height=1200)

    fig.add_trace(go.Scatter(x=silo_data["AcquisitionTime"].values, y=silo_data["perc_filled"], name=r"% filled", fill='tozeroy', fillcolor="rgba(0, 0, 255, 0.1)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=resampled_silo_data["AcquisitionTime"].values, y=resampled_silo_data["filtered_perc1"], name=r"% reg", fill='tozeroy', fillcolor="rgba(255, 0, 0, 0.1)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=resampled_silo_data["AcquisitionTime"].values, y=resampled_silo_data["filtered_perc2"], name=r"% savgol", fill='tozeroy', fillcolor="rgba(0, 255, 0, 0.1)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=resampled_silo_data["AcquisitionTime"].values, y=resampled_silo_data["filtered_perc3"], name=r"% savgol non-real-time", fill='tozeroy', fillcolor="rgba(255, 255, 0, 0.1)"), row=1, col=1)
    fig.update_yaxes(title_text=r"% Silo filled", range=[-1, 101], row=1, col=1)

    fig.add_trace(go.Scatter(x=silo_data["AcquisitionTime"].values, y=silo_data["DistanceFO"], name="Distance FO"), row=2, col=1)
    fig.add_trace(go.Scatter(x=silo_data["AcquisitionTime"].values, y=silo_data["DistanceCDM"], name="Distance CDM"), row=2, col=1)
    fig.update_yaxes(title_text="Distance [m]", range=[-0.5, 10.5], row=2, col=1)

    #fig.add_trace(go.Scatter(x=silo_data["AcquisitionTime"].values, y=silo_data["cdm_weight"], name="CDM Weight"), row=3, col=1)
    #fig.update_yaxes(title_text="CDM Weight", range=[-0.1, 1.1], row=3, col=1)

    #fig.update_xaxes(title_text="Date", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)
    """
    fig1 = px.line(
        silo_data,
        x="AcquisitionTime",
        y=["perc_filled", "filtered_perc1", "filtered_perc2"],
        template=plotly_theme,
        height=400
    )
    fig1.update_layout(yaxis_range=[0, 100], xaxis_title="Date", yaxis_title=r"% Silo filled")

    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.line(
        silo_data,
        x="AcquisitionTime",
        y=["DistanceFO", "DistanceCDM"],
        template=plotly_theme,
        height=400
    )
    fig2.update_layout(yaxis_range=[0, 10], xaxis_title="Date", yaxis_title="Distance [m]")

    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.line(
        silo_data,
        x="AcquisitionTime",
        y="cdm_weight",
        template=plotly_theme,
        height=400
    )
    fig3['data'][0]['showlegend'] = True
    fig3['data'][0]['name'] = 'CDM Weight'
    fig3.update_layout(yaxis_range=[-0.1, 1.1], xaxis_title="Date", yaxis_title="CDM Weight")

    st.plotly_chart(fig3, use_container_width=True)
    """


def plot_silo_matplotlib(data, silo_name):
    x = data[data["LocationName"] == silo_name]["AcquisitionTime"]
    y = data[data["LocationName"] == silo_name]["DistanceFO"]
    plt.plot(x, y)
    plt.title(f"Silo : {silo_name}")
    plt.show()

