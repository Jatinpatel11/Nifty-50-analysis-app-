import streamlit as st

st.set_page_config(
    page_title="Nifty 50 Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Nifty 50 Analysis Dashboard - Streamlit Cloud Deployment")

st.markdown("""
## Welcome to the Nifty 50 Analysis Dashboard!

This is a permanent deployment on Streamlit Cloud. The full application will load momentarily.

Please wait while we initialize the dashboard...
""")

st.info("Loading the main application... Please refer to app_without_price_analysis.py for the full dashboard.")

# Redirect to the main application
import app.py
