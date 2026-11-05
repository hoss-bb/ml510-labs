"""Temporary predictions; no stored listing is edited."""
import pandas as pd
import streamlit as st
from .common import request, money

ROOMS = ["Entire home/apt", "Private room", "Shared room", "Hotel room"]

def scenario_page(client, detail=None, embedded=False):
    if embedded:
        # The property workspace supplies its source explicitly.
        source = detail["listing"]
        identity = source["listing_id"]
        st.caption("Try different property inputs without changing this stored listing.")
        st.info(f"Starting from {source['property_type']} · {source['neighbourhood']} · Listing {identity}.")
    else:
        detail = None
        identity = "custom"
        source = {"accommodates": None, "bedrooms": None, "bathrooms": None, "beds": None,
                  "neighbourhood": "", "property_type": "", "room_type": None}
        st.title("Price scenario")
        st.write("Have property details you want to explore? Enter them below to estimate an advertised nightly price in Montréal, in CAD. Change the inputs to compare scenarios.")
        st.caption("This is an independent simulation. It does not create or update a listing. The estimate is not a recommended price or a revenue forecast.")
    def input_key(field):
        return f"scenario_input_{identity}_{field}"
    for field in ("accommodates", "bedrooms", "bathrooms", "beds", "neighbourhood", "property_type", "room_type"):
        value = source[field]
        if field in ("bedrooms", "bathrooms", "beds") and value is not None:
            value = float(value)
        st.session_state.setdefault(input_key(field), value)
    # Inputs rerun locally so tab changes retain edits; only Calculate calls the API.
    with st.container(border=True):
        with st.container():
            st.subheader("Property")
            values = {"accommodates": st.number_input("Guests", min_value=1, max_value=100, value=None, step=1, key=input_key("accommodates"))}
            for field in ("bedrooms", "bathrooms", "beds"):
                values[field] = st.number_input(field.capitalize(), min_value=0.0,
                                                value=None, step=0.5, key=input_key(field),
                                                help="Leave blank if unknown; the model uses its fitted missing-value handling.")
        with st.container():
            st.subheader("Location and type")
            values["neighbourhood"] = st.text_input("Neighbourhood", key=input_key("neighbourhood"), placeholder="Enter the Montréal neighbourhood", max_chars=200).strip()
            values["property_type"] = st.text_input("Property type", key=input_key("property_type"), placeholder="For example: Entire rental unit", max_chars=200).strip()
            values["room_type"] = st.selectbox("Room type", ROOMS, index=None, key=input_key("room_type"))
        ready = values["accommodates"] is not None and all(values[field] for field in ("neighbourhood", "property_type", "room_type"))
        if not ready:
            st.caption("Enter guests, neighbourhood and property type, then select a room type. Bedrooms, bathrooms and beds can be left blank if unknown.")
        submitted = st.button("Calculate estimate", type="primary", key=f"calculate_{identity}", disabled=not ready)
    if submitted:
        with st.spinner("Calculating estimate…"):
            result = request(client, "POST", "/predict", json=values)
        # A failed submission must not leave an old estimate looking like its result.
        st.session_state[f"scenario_result_{identity}"] = {"prediction": result, "inputs": values} if result else None
    saved = st.session_state.get(f"scenario_result_{identity}")
    if saved:
        result = saved["prediction"]
        with st.container(border=True):
            st.subheader("Last calculated estimate")
            st.caption("Edits above take effect only after Calculate estimate.")
            if result["estimate"] > 0:
                st.metric("Estimated nightly price", money(result["estimate"]))
                if detail and detail["prediction"]["estimate"] > 0:
                    difference = result["estimate"] - detail["prediction"]["estimate"]
                    st.caption(f"{difference:+,.0f} CAD compared with the original estimate. This is a model comparison, not a guaranteed effect on price.")
            else:
                st.warning("Estimate requires review: the model returned a nonpositive value.")
            with st.expander("Submitted inputs and model"):
                st.dataframe(pd.DataFrame([saved["inputs"]]).rename(columns=lambda name: name.replace("_", " ").capitalize()),
                             hide_index=True, width="stretch")
                st.caption("Model: " + result["model_version"])
                st.caption("Request: " + result["request_id"])
