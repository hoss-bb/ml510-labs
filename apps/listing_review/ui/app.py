"""Streamlit entry point: configuration, navigation and page dispatch."""
import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from apps.listing_review.ui.helpers.api_client import ApiClient
from apps.listing_review.ui.helpers.listings import listings_page
from apps.listing_review.ui.helpers.scenario import scenario_page
from apps.listing_review.ui.helpers.arrivals import arrivals_page

app_directory = Path(__file__).resolve().parents[1]
load_dotenv(app_directory / ".env", override=False)
load_dotenv(app_directory.parents[1] / ".env", override=False)
client = ApiClient(os.getenv("S10_API_URL", "http://127.0.0.1:8000"))
st.set_page_config(page_title="Montréal | Listing review", page_icon="🏘️", layout="wide")

# Retain local widget state when its tab or page is hidden.
for key in list(st.session_state):
    if key.startswith(("scenario_input_", "quality_draft_", "property_tab_", "filter_")):
        st.session_state[key] = st.session_state[key]

with st.sidebar:
    st.title("Listing review")
    st.caption("MONTRÉAL · CAD")
    st.radio("Workspace", ["Listings", "Price scenario", "Arrivals"], key="view")
    st.divider()
    st.caption("Advertised-price estimates and AI review proposals for a manager’s inspection.")

{"Listings": listings_page, "Price scenario": scenario_page, "Arrivals": arrivals_page}[st.session_state["view"]](client)
