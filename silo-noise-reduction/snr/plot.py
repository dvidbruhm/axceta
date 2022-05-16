import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import numpy as np

from config.config import logger
from snr.filters import realtime_savgol
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

    silo_name = st.selectbox("Silo list", options=locations)
    st.subheader(f"Silo name : {silo_name}")

    silo_data = batch_dist_to_vol(data, conversion_data, silo_name)

    silo_data["filtered_perc1"] = savgol_filter(silo_data["perc_filled"].values, 11, 1)
    window_len = 30
    silo_data["filtered_perc2"] = np.convolve(silo_data["perc_filled"].values, np.ones(window_len), 'same') / window_len
    #silo_data["realtime_filtered_dist"] = realtime_savgol(silo_data["DistanceFO"])

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
    fig3.update_layout(yaxis_range=[-0.1, 1.1], xaxis_title="Date", yaxis_title="CDM Weight")

    st.plotly_chart(fig3, use_container_width=True)
    


def plot_silo_matplotlib(data, silo_name):
    x = data[data["LocationName"] == silo_name]["AcquisitionTime"]
    y = data[data["LocationName"] == silo_name]["DistanceFO"]
    plt.plot(x, y)
    plt.title(f"Silo : {silo_name}")
    plt.show()

