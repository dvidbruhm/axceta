import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.signal import savgol_filter

from config.config import logger
from snr.filters import realtime_savgol
from snr.utils import dist_to_volume


def plot_data_streamlit(data: pd.DataFrame, conversion_data: pd.DataFrame) -> None:
    logger.info("Starting streamlit plot...")
    st.title('Filters experimentation on silo data')

    if st.checkbox("Show raw data", value=False):
        st.subheader("Raw silo data")
        st.write(data)

    locations = data["LocationName"].unique()
    locations = ["Jacobs-001"]

    location = st.selectbox("Silo list", options=locations)
    st.subheader(f"Silo name : {location}")

    silo_data = data[data["LocationName"] == location].copy()

    plotly_theme = "plotly_dark"
    #silo_data["filtered_dist"] = savgol_filter(silo_data["DistanceFO"].values, 11, 2)
    #silo_data["realtime_filtered_dist"] = realtime_savgol(silo_data["DistanceFO"])
    print(silo_data["DistanceFO"])
    """
    fig1 = px.line(
        silo_data,
        x="AcquisitionTime",
        y="perc_filled",
        template=plotly_theme,
        height=400,
        labels={
            "perc_filled": r"% silo filled",
            "AcquisitionTime": "Date time"
        }
    )

    st.plotly_chart(fig1, use_container_width=True)
    """
    fig2 = px.line(
        silo_data,
        x="AcquisitionTime",
        y="DistanceFO",
        template=plotly_theme,
        height=400,
        labels={
            "DistanceFO": "Distance [m]",
            "AcquisitionTime": "Date time"
        }
    )

    st.plotly_chart(fig2, use_container_width=True)
