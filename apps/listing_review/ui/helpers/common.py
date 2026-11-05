"""Small presentation helpers shared by the workspace pages."""
import html
import re
import streamlit as st
from .api_client import ServiceError


def request(client, method, path, **kwargs):
    try:
        return client.request(method, path, **kwargs)
    except ServiceError as error:
        st.error(str(error))
        if error.request_id:
            st.caption("Support reference: " + error.request_id)
        return None


def money(value):
    return f"${value:,.0f} CAD"


def open_scenario():
    st.session_state["view"] = "Price scenario"


def text_description(value):
    # Source descriptions may contain HTML. Display text, never executable markup.
    return html.unescape(re.sub(r"<[^>]+>", " ", value)).strip()



STATUS = {"not_requested": "Not requested", "queued": "Queued", "running": "In progress",
          "completed": "Completed", "failed": "Needs retry"}


def collection(client, path):
    """Read bounded API pages without truncating counts or historical reviews."""
    rows = []
    while True:
        page = request(client, "GET", path, params={"offset": len(rows), "limit": 250})
        if page is None:
            return None
        rows.extend(page)
        if len(page) < 250:
            return rows
