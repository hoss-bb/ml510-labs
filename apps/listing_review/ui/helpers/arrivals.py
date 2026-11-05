"""Batch progress and rejected records."""
from datetime import datetime, timezone
import pandas as pd
import streamlit as st
from .common import collection, request

def arrivals_page(client):
    st.title("Arrivals")
    st.caption("Follow incoming batches and check which records were accepted.")
    st.button("Refresh arrivals")
    batches = collection(client, "/batches")
    if batches is None:
        return
    st.caption("Data refreshed: " + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
    st.caption("Saved processing results; this does not indicate whether the publisher or ingestion service is currently running.")
    st.write(f"{len(batches)} batches · {sum(row['accepted'] for row in batches)} accepted records · {sum(row['rejected'] for row in batches)} rejected records")
    if not batches:
        st.info("No batches received yet. Completed arrivals will appear here automatically when processed.")
        return
    st.dataframe(pd.DataFrame([{"Batch": row["id"], "Status": row["status"].capitalize(),
                               "Accepted": row["accepted"], "Rejected": row["rejected"],
                               "Last updated · UTC": row["updated"]} for row in batches]),
                 hide_index=True, width="stretch")
    rejected_batches = [row for row in batches if row["rejected"]]
    if rejected_batches:
        st.subheader("Records needing attention")
        batch_id = st.selectbox("Batch with rejected records", [row["id"] for row in rejected_batches])
        rejected = collection(client, f"/batches/{batch_id}/rejections")
        if rejected:
            st.dataframe(pd.DataFrame([{"Row": row["row_number"], "Listing": row["listing_id"],
                                       "Reason": row["reason"].replace("_", " ").capitalize()} for row in rejected]),
                         hide_index=True, width="stretch")
    elif all(row["status"] == "completed" for row in batches):
        st.success("All received records were accepted.")
    for row in batches:
        if row["status"] == "failed":
            st.warning(f"Batch {row['id']} did not finish. Previously saved listings remain available.")
    with st.expander("Processing details"):
        st.dataframe(pd.DataFrame([{"Batch": row["id"], "Seconds": row["seconds"], "Checksum": row["checksum"],
                                   "Failure": row["error"]} for row in batches]), hide_index=True, width="stretch")
        st.caption("Batch numbers represent simulated arrivals, not original publication dates.")
