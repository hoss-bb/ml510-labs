"""Inventory selection and a dedicated property workspace."""
import streamlit as st
from .common import request, money, STATUS, text_description
from .reviews import review_panel
from .scenario import scenario_page

PAGE_SIZE = 5


def back_to_listings():
    st.session_state["listing_detail_open"] = False
    st.session_state["table_revision"] = st.session_state.get("table_revision", 0) + 1


def clear_filters():
    for field in ("neighbourhood", "room_type", "batch_id", "status", "search"):
        st.session_state[f"filter_{field}"] = ""
    st.session_state.update(inventory_filters={}, inventory_page=0, selected_listing=None)
    back_to_listings()


def listing_detail(client, listing_id):
    st.button("Back to listings", on_click=back_to_listings)
    detail = request(client, "GET", f"/listings/{listing_id}")
    if detail is None:
        st.button("Retry loading property")
        return
    listing, prediction = detail["listing"], detail["prediction"]
    st.header(listing["property_type"])
    st.write(listing["neighbourhood"])
    st.caption(f"Listing {listing_id} · {detail['batch_id']} · {listing['room_type']}")
    overview, review, scenario = st.tabs(
        ["Overview", "Quality review", "Price scenario"],
        key=f"property_tab_{listing_id}", on_change="rerun")
    if overview.open:
        with overview:
            with st.container(border=True):
                st.subheader("Property facts")
                st.write(" · ".join(f"{label}: {'Not supplied' if listing[field] is None else f'{listing[field]:g}'}"
                    for field, label in [("accommodates", "Guests"), ("bedrooms", "Bedrooms"),
                                         ("beds", "Beds"), ("bathrooms", "Bathrooms")]))
                with st.expander("Listing description"):
                    st.text(text_description(listing["description"]) or "No description supplied.")
            with st.container(border=True):
                st.subheader("Nightly price · CAD")
                a, b = st.columns(2)
                a.metric("Advertised", money(listing["advertised_price"]))
                if prediction["estimate"] > 0:
                    b.metric("Model estimate", money(prediction["estimate"]))
                    difference = listing["advertised_price"] - prediction["estimate"]
                    st.caption(f"Advertised minus estimated: {difference:+,.0f} CAD. A difference is a reason to investigate, not proof of incorrect pricing.")
                else:
                    b.metric("Model estimate", "Requires review")
                    st.warning("The model returned a nonpositive estimate. Do not use it as a nightly price.")
                with st.expander("Estimate details"):
                    st.caption("Model version: " + prediction["model_version"])
                    st.caption("An advertised-price estimate, not an optimal price or expected revenue.")
    if review.open:
        with review:
            review_panel(client, listing_id)
    if scenario.open:
        with scenario:
            scenario_page(client, detail=detail, embedded=True)


def listings_page(client):
    if st.session_state.get("listing_detail_open") and st.session_state.get("selected_listing"):
        listing_detail(client, st.session_state["selected_listing"])
        return
    st.title("Listings")
    st.caption("Select a property to compare prices, inspect its quality review or try a temporary scenario.")
    filters = st.session_state.get("inventory_filters", {})
    page = st.session_state.get("inventory_page", 0)
    data = request(client, "GET", "/listings/workspace", params={**filters, "offset": page * PAGE_SIZE, "limit": PAGE_SIZE})
    if data is None:
        st.button("Retry loading listings")
        return
    # Counts can shrink while a review changes status or a batch is processed.
    last_page = max(0, (data["total"] - 1) // PAGE_SIZE)
    if page > last_page:
        st.session_state["inventory_page"] = last_page
        st.session_state["table_revision"] = st.session_state.get("table_revision", 0) + 1
        st.rerun()
    summary = data["summary"]
    st.caption(f"{summary['listings']} received listings · {summary['in_progress']} reviews in progress · {summary['completed']} completed reviews (latest attempts, all listings)")
    selected = {}
    st.session_state.setdefault("filter_search", filters.get("search", ""))
    selected["search"] = st.text_input("Search listing ID", placeholder="Full ID or part of an ID", key="filter_search").strip()
    active = [f"{key.replace('_', ' ').capitalize()}: {STATUS.get(value, value) if key == 'review_status' else value}"
              for key, value in filters.items() if key != "search"]
    with st.expander("Filters" + (" · " + " · ".join(active) if active else " · All categories")):
        cols = [*st.columns(2), *st.columns(2)]
        for col, field, label, all_label in zip(cols[:3], ["neighbourhood", "room_type", "batch_id"],
                ["Neighbourhood", "Room type", "Arrival batch"], ["All neighbourhoods", "All room types", "All batches"]):
            st.session_state.setdefault(f"filter_{field}", filters.get(field, ""))
            selected[field] = col.selectbox(label, ["", *data["options"][field]],
                format_func=lambda value, default=all_label: value or default, key=f"filter_{field}")
        st.session_state.setdefault("filter_status", filters.get("review_status", ""))
        selected["review_status"] = cols[3].selectbox("Review status", ["", *STATUS],
            format_func=lambda value: STATUS[value] if value else "All statuses", key="filter_status")
    with st.container(horizontal=True):
        st.button("Clear filters", on_click=clear_filters, disabled=not filters)
        st.button("Refresh listings")
    selected = {key: value for key, value in selected.items() if value}
    if selected != filters:
        st.session_state.update(inventory_filters=selected, inventory_page=0, selected_listing=None)
        st.session_state["table_revision"] = st.session_state.get("table_revision", 0) + 1
        st.rerun()
    if not data["items"]:
        st.info("No listings match these filters. Clear filters or try another ID." if summary["listings"] else
                "No arrivals yet. Listings will appear here after the next batch is processed.")
        return
    ids = tuple(item["listing"]["listing_id"] for item in data["items"])
    suffixes = [identifier[-8:] for identifier in ids]
    # Short references aid scanning; resolve a visible suffix collision with the full ID.
    references = ["…" + identifier[-8:] if len(identifier) > 8 and suffixes.count(identifier[-8:]) == 1
                  else identifier for identifier in ids]
    def open_listing(identifier):
        st.session_state.update(selected_listing=identifier, listing_detail_open=True)

    st.caption("Nightly prices in CAD. References show the last eight ID digits; full IDs and descriptions are in the details.")
    for reference, item in zip(references, data["items"]):
        listing, prediction = item["listing"], item["prediction"]
        with st.container(border=True):
            facts, prices, action = st.columns([2, 2, 1], vertical_alignment="center")
            with facts:
                st.write(f"**{reference}**")
                bedrooms = "Bedrooms not supplied" if listing["bedrooms"] is None else f"{listing['bedrooms']:g} bedrooms"
                st.caption(f"{listing['accommodates']} guests · {bedrooms}")
            with prices:
                st.write("Advertised: " + money(listing["advertised_price"]))
                estimate = money(prediction["estimate"]) if prediction["estimate"] > 0 else "Requires review"
                st.write("Estimate: " + estimate)
                st.caption("Review: " + STATUS[item["review_status"]])
            action.button("", icon=":material/arrow_forward:", help=f"Open listing details · {listing['listing_id']}",
                key=f"open_listing_{listing['listing_id']}", on_click=open_listing, args=(listing["listing_id"],))
    start = page * PAGE_SIZE
    st.caption(f"Showing {start + 1}–{start + len(data["items"])} of {data['total']} matching listings · Page {page + 1} of {last_page + 1}")
    def change_page(target):
        st.session_state.update(inventory_page=target, selected_listing=None)
    with st.container(horizontal=True):
        st.button("Previous", disabled=page == 0, on_click=change_page, args=(page - 1,))
        st.button("Next", disabled=page == last_page, on_click=change_page, args=(page + 1,))
