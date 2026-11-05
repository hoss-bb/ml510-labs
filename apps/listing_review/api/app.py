"""HTTP contracts; model work stays in the shared prediction component."""
from contextlib import asynccontextmanager
from .helpers.http import register_http_handlers
from .helpers.reviews import process_review
from typing import Literal
from .schemas import (LiveResponse, ReadyResponse, StoredListing, WorkspaceResponse, BatchResponse,
                      RejectionResponse, ReviewAccepted, ReviewSummary, JobResponse, ErrorResponse)
import sqlite3
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request, Response, Body
from lab_helpers.s10_shared.prediction.schemas import Features, PredictionResponse
from lab_helpers.s10_shared.reviews.schemas import ReviewRequest
from lab_helpers.s10_shared.runtime.settings import settings
from lab_helpers.s10_shared.runtime.storage import Store, Conflict
from lab_helpers.s10_shared.reviews.context import ContextCatalog
from lab_helpers.s10_shared.prediction.service import Predictor
from lab_helpers.s10_shared.reviews.prompts import load_prompts, PromptError
from lab_helpers.s10_shared.reviews.configuration import generation_settings
from lab_helpers.s10_shared.runtime.logging import event


def create_app(config=None, predictor=None):
    config = config or settings()
    store = Store(config.database)

    @asynccontextmanager
    async def lifespan(app):
        store.initialize()
        interrupted = store.recover_reviews()
        event("review_recovery", outcome="completed", interrupted_jobs=interrupted)
        app.state.review_context = ContextCatalog(config.data / "prepared" / "review_context.json")
        app.state.predictor = predictor
        if app.state.predictor is None:
            try:
                app.state.predictor = Predictor(config.model)
            except FileNotFoundError:
                event("startup", outcome="model_unavailable")
        yield

    app = FastAPI(title="S10 listing review", version="1.0", lifespan=lifespan,
        description="Local teaching API: synchronous price estimates and persistent asynchronous review jobs. "
                    "Run one API process. Interrupted reviews require an explicit new attempt.",
        responses={status: {"model": ErrorResponse, "description": description} for status, description in {
            404: "Resource not found", 405: "Method not allowed", 409: "Conflicting idempotency key", 422: "Invalid request fields",
            500: "Unexpected internal failure", 503: "Required service unavailable"}.items()})

    register_http_handlers(app)

    @app.get("/health/live", response_model=LiveResponse, tags=["Health"],
             description="Check process liveness.")
    def live():
        return {"status": "ok"}

    @app.get("/health/ready", response_model=ReadyResponse, tags=["Health"],
             description="Check model and storage readiness.")
    def ready():
        if app.state.predictor is None:
            raise HTTPException(503, "Model unavailable; prepare a trusted model and restart.")
        try:
            store.check_ready()
        except sqlite3.Error:
            raise HTTPException(503, "Storage unavailable") from None
        return {"status": "ready", "model_version": app.state.predictor.version}

    @app.post("/predict", response_model=PredictionResponse, tags=["Predictions"],
              description="Estimate one advertised nightly price in CAD immediately; stores nothing. Nonpositive estimates require review.")
    def predict(request: Request, record: Features = Body(openapi_examples={"apartment": {
        "summary": "A four-guest apartment", "value": {"accommodates": 4, "bedrooms": 2,
        "bathrooms": 1, "beds": 2, "property_type": "Entire rental unit",
        "room_type": "Entire home/apt", "neighbourhood": "Ville-Marie"}}})):
        if app.state.predictor is None:
            raise HTTPException(503, "Model unavailable")
        result = app.state.predictor.predict([record])[0]
        return PredictionResponse(**result.model_dump(), request_id=request.state.request_id)

    @app.get("/batches", response_model=list[BatchResponse], tags=["Arrivals"],
             description="Saved processing results ordered by batch ID. Offset pagination; an empty page ends the collection.")
    def batches(offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=250)):
        return store.batches(offset=offset, limit=limit)

    @app.get("/batches/{batch_id}/rejections", response_model=list[RejectionResponse], tags=["Arrivals"],
             description="Rejected rows ordered by source row number. No rejections returns []; unknown batch returns 404.")
    def rejections(batch_id: str, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=250)):
        if store.batch(batch_id) is None:
            raise HTTPException(404, "Unknown batch")
        return store.rejections(batch_id, offset=offset, limit=limit)

    @app.get("/listings", response_model=list[StoredListing], tags=["Listings"],
             description="Stored facts and batch estimates, ordered by batch ID then listing ID. Reads never recalculate prices.")
    def listings(offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=250)):
        return store.listings(offset=offset, limit=limit)

    @app.get("/listings/workspace", response_model=WorkspaceResponse, tags=["Listings"],
             description="Filter before pagination. Total is the matching count; summary and options cover all listings. Ordered by newest batch ID, then listing ID.")
    def workspace(offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=250),
                  neighbourhood: str | None = None, room_type: str | None = None,
                  batch_id: str | None = None,
                  review_status: Literal["not_requested", "queued", "running", "completed", "failed"] | None = None,
                  search: str = Query("", max_length=100)):
        return store.workspace(offset=offset, limit=limit, neighbourhood=neighbourhood,
                               room_type=room_type, batch_id=batch_id,
                               review_status=review_status, search=search)

    @app.get("/listings/{listing_id}", response_model=StoredListing, tags=["Listings"],
             description="Read stored property facts and batch estimate.")
    def listing(listing_id: str):
        try:
            return store.listing(listing_id)
        except KeyError:
            raise HTTPException(404, "Unknown listing") from None

    @app.post("/listings/{listing_id}/reviews", status_code=202, response_model=ReviewAccepted, tags=["Reviews"],
              description="Submit an asynchronous review. Reuse request_id for a network retry: the same listing returns the same job, even after completion/failure. Another listing with that key returns 409. Send a new key for an explicit new attempt. Poll status_url; polling never submits work.")
    def request_review(listing_id: str, request: ReviewRequest, background_tasks: BackgroundTasks, response: Response):
        existing = store.job_for_request(request.request_id)
        if existing is not None:
            if existing["listing_id"] != listing_id:
                raise HTTPException(409, "Request ID already belongs to another listing.")
            job_id = existing["id"]
            response.headers["Location"] = f"/jobs/{job_id}"
            return {"job_id": job_id, "status_url": f"/jobs/{job_id}"}
        try:
            store.listing(listing_id)
            instructions = load_prompts()
            try:
                generation = generation_settings()
            except ValueError:
                raise HTTPException(503, "AI configuration unavailable; check provider and model settings.") from None
            job_id = store.create_job(listing_id, request.request_id, instructions, generation, app.state.review_context)
            background_tasks.add_task(process_review, store, job_id)
            response.headers["Location"] = f"/jobs/{job_id}"
            return {"job_id": job_id, "status_url": f"/jobs/{job_id}"}
        except KeyError:
            raise HTTPException(404, "Unknown listing") from None
        except PromptError:
            raise HTTPException(503, "Review prompts unavailable; supply all three instruction files.") from None
        except Conflict as error:
            raise HTTPException(409, str(error)) from None

    @app.get("/jobs/{job_id}", response_model=JobResponse, tags=["Reviews"],
             description="Poll queued/running/completed/failed status. A failed job still returns HTTP 200. Historical outputs are preserved.")
    def job(job_id: str):
        try:
            return store.job(job_id)
        except KeyError:
            raise HTTPException(404, "Unknown job") from None

    @app.get("/listings/{listing_id}/reviews", response_model=list[ReviewSummary], tags=["Reviews"],
             description="Review attempts newest first. Unknown listing returns 404; no attempts returns []. Use offset/limit for older history.")
    def listing_reviews(listing_id: str, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=250)):
        try:
            store.listing(listing_id)
        except KeyError:
            raise HTTPException(404, "Unknown listing") from None
        return store.listing_reviews(listing_id, offset=offset, limit=limit)

    return app


app = create_app()
