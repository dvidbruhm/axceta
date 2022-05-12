import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.signal import savgol_filter

from config.config import logger
from snr.filters import realtime_savgol


def plot_data_streamlit(data: pd.DataFrame) -> None:
    logger.info("Starting streamlit plot...")
    st.title('Filters experimentation on silo data')

    if st.checkbox("Show raw data", value=False):
        st.subheader("Raw silo data")
        st.write(data)

    locations = data["LocationName"].unique()

    location = st.selectbox("Silo list", options=locations)
    st.subheader(f"Silo name : {location}")

    silo_data = data[data["LocationName"] == location].copy()
    plotly_theme = "plotly_dark"
    silo_data["filtered_dist"] = savgol_filter(silo_data["DistanceFO"].values, 11, 2)
    silo_data["realtime_filtered_dist"] = realtime_savgol(silo_data["DistanceFO"])

    fig = px.line(
        silo_data,
        x="AcquisitionTime",
        y=["DistanceFO", "filtered_dist"],
        template=plotly_theme,
        height=800,
        labels={
            "DistanceFO": "Distance [m]",
            "AcquisitionTime": "Date time"
        }
    )

    st.plotly_chart(fig, use_container_width=True)
