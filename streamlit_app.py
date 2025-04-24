import streamlit as st
import pandas as pd
import numpy as np
# Import functions from your secondary file
from app_without_price_analysis import display_dashboard, load_data, analyze_data

# Set page config ONCE at the very top before any other Streamlit commands
st.set_page_config(
    page_title="Nifty 50 Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main app code
st.title("Dashboard - Streamlit Cloud Deployment")
st.header("Welcome to the Nifty 50 Analysis Dashboard!")
st.write("This is a permanent deployment on Streamlit Cloud.")

# Load data
data = load_data()

# Display the dashboard components
display_dashboard(data)

# Add any additional main app functionality here
if st.sidebar.checkbox("Show Analysis"):
    st.subheader("Data Analysis")
    analyzed_data = analyze_data(data)
    st.write(analyzed_data)
