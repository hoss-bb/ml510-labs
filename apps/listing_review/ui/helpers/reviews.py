"""Review actions, polling and current/legacy result presentation."""
import uuid
import pandas as pd
import streamlit as st
from .common import collection, request

from .common import STATUS
from lab_helpers.s10_shared.reviews.messages import REVIEW_ERRORS

def show_quality_proposal(job):
    result = job["result"]
    labels = {"confirm_facts": "Confirm conflicting facts", "clarify_condition": "Clarify an important condition",
              "guest_note": "Prepare a guest-information note", "no_change": "No change indicated"}
    st.subheader(labels[result["action"]])
    st.markdown("**What we found**")
    st.write(result["reason"])
    st.markdown("**Suggested next step**")
    st.write(result["next_step"])
    if result["confirm_before_use"]:
        st.markdown("**Confirm first**")
        for fact in result["confirm_before_use"]:
            st.write("• " + fact)
    draft = result["owner_question"] or result["guest_note"]
    if draft:
        label = "Owner-question draft" if result["owner_question"] else "Before you book · draft"
        if result["guest_note"]:
            st.info("Optional wording for manager review, not a detected defect or a published change.")
        st.session_state.setdefault(f"quality_draft_{job['id']}", draft)
        edited = st.text_area(label, height=140, key=f"quality_draft_{job['id']}")
        st.download_button("Download draft", edited, file_name=f"listing-{job['listing_id']}-draft.txt",
                           mime="text/plain", key=f"quality_download_{job['id']}")
        st.caption("Your edits affect this download only. Nothing is sent, published or changed in the listing.")
    evidence = {item["id"]: item for item in job["stages"]["evidence"]["output"]["items"]}
    if result["evidence_ids"]:
        with st.expander("Supporting facts"):
            groups = {}
            for identifier in result["evidence_ids"]:
                item = evidence[identifier]
                groups.setdefault((item["source_id"], item["excerpt"]), []).append(item)
            for (source, excerpt), items in groups.items():
                for item in items:
                    st.write(item["id"] + " · " + item["claim"])
                labels = {"listing.description": "Listing description", "listing.accommodates": "Listed guest capacity",
                          "context.bathroom_description": "Bathroom information", "context.minimum_nights": "Minimum nights"}
                label = labels.get(source, source.split(".")[-1].replace("_", " ").capitalize())
                st.caption("Listed amenity" if source.startswith("context.amenities.") else label)
                st.text(excerpt)
    context = job["snapshot"].get("review_context", {})
    provenance = context.get("provenance")
    if provenance:
        st.caption("Additional context: Inside Airbnb snapshot " + provenance["snapshot"] + ". Owner confirmation may still be needed.")
    else:
        st.caption("No matching additional context was available; this review uses the supplied listing facts.")


def show_proposal(job):
    result = job["result"]
    if result.get("output_version") == 2:
        show_quality_proposal(job)
        return
    st.caption("Earlier review format · retained for reference")
    labels = {"no_issue_identified": "No issue identified", "clarification_needed": "Clarification suggested",
              "review_recommended": "Review recommended"}
    st.caption(labels[result["review_status"]])
    st.subheader(result["recommended_action"])
    st.write(result["justification"])
    evidence = job["stages"].get("evidence", {}).get("output", {}).get("items", [])
    with st.expander("Supporting evidence"):
        if not result["evidence_indices"]:
            st.caption("No description excerpts cited in this recommendation.")
        for index in result["evidence_indices"]:
            item = evidence[index]
            st.write(item["claim"])
            st.text('“' + item["excerpt"] + '”')
            if item.get("structured_field"):
                st.caption("Compared with: " + item["structured_field"].replace("_", " "))
    if result["questions"]:
        st.markdown("**Questions to resolve**")
        for question in result["questions"]:
            st.write("• " + question)
    if result["pricing_caveat"]:
        st.info(result["pricing_caveat"])
    with st.expander("Draft message to the owner"):
        st.text(result["owner_message"])
        st.download_button("Download draft", result["owner_message"],
                           file_name=f"listing-{job['listing_id']}-draft.txt", mime="text/plain",
                           key=f"draft_{job['id']}")
        st.caption("Draft only. Nothing is sent to the owner.")


def show_job(job):
    if job["status"] == "failed":
        st.warning(REVIEW_ERRORS.get(job["error"], "The AI service could not finish this review. Try again later; the price estimate remains available."))
    elif job["status"] in {"queued", "running"}:
        st.info("Waiting for the background review to start…" if job["status"] == "queued" else
                "Review in progress. Completed stages appear below.")
    elif job["status"] == "completed":
        st.caption("Review completed · proposal for manager inspection")
    if job["result"]:
        show_proposal(job)
    with st.expander("Technical details"):
        st.caption(f"Job: {job['id']} · Created: {job['created']} UTC")
        st.caption("Status: " + STATUS[job["status"]])
        if job["error"]:
            st.code(job["error"], language=None)
        if job["stages"]:
            st.dataframe(pd.DataFrame([
                {"Stage": name.capitalize(), "Seconds": round(stage.get("seconds", 0), 2)}
                for name, stage in job["stages"].items()]), hide_index=True, width="stretch")
        if job["snapshot"].get("prompt_version"):
            st.caption("Historical prompt version: " + job["snapshot"]["prompt_version"])


def review_panel(client, listing_id):
    jobs = collection(client, f"/listings/{listing_id}/reviews")
    if jobs is None:
        return
    latest = jobs[0] if jobs else None
    active = latest is not None and latest["status"] in {"queued", "running"}
    label = "Generate review" if latest is None else (
        "Retry review" if latest["status"] == "failed" else "Generate another review")
    # An attempt retains its token across network errors and UI reruns.
    attempt_key = f"attempt_{listing_id}_{latest['id'] if latest else 'first'}"
    if st.button(label, type="secondary" if latest and latest["status"] == "completed" else "primary", disabled=active, key=f"review_{listing_id}"):
        token = st.session_state.setdefault(attempt_key, uuid.uuid4().hex)
        with st.spinner("Requesting review…"):
            created = request(client, "POST", f"/listings/{listing_id}/reviews", json={"request_id": token})
        if created:
            st.rerun()
    st.caption("One evidence-based next step. The manager decides whether to use it.")
    if not latest:
        st.info("No quality review yet. Check for conflicting facts, clarify an important condition, prepare a guest note, or leave the listing unchanged.")
        return

    @st.fragment(run_every="3s" if active else None)
    def current_review():
        job = request(client, "GET", f"/jobs/{latest['id']}")
        if job is not None:
            if job["status"] != latest["status"]:
                st.rerun()
            show_job(job)
    current_review()
    if len(jobs) > 1:
        with st.expander("Previous reviews"):
            choices = {job["id"]: f"{job['created']} UTC · {STATUS[job['status']]} · attempt {len(jobs)-i}"
                       for i, job in enumerate(jobs)}
            previous = st.selectbox("Review history", [job["id"] for job in jobs[1:]],
                                    format_func=choices.get, key=f"history_{listing_id}")
            job = request(client, "GET", f"/jobs/{previous}")
            if job is not None:
                show_job(job)
