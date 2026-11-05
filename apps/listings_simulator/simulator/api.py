"""A separate upstream platform: arrival files become immutable batches."""
import hashlib
import re
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from .helpers.settings import settings


def create_app(incoming=None):
    folder = incoming or settings().incoming
    app = FastAPI(title="S10 upstream listing platform")

    @app.get("/health/live")
    def live():
        return {"status": "ok"}

    @app.get("/batches")
    def batches():
        if not folder.is_dir():
            return []
        return [{"id": p.stem, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in sorted(folder.glob("*.csv")) if re.fullmatch(r"day_[0-9]+", p.stem)]

    @app.get("/batches/{batch_id}")
    def batch(batch_id: str):
        if not re.fullmatch(r"day_[0-9]+", batch_id):
            raise HTTPException(404, "Unknown batch")
        path = folder / f"{batch_id}.csv"
        if not path.is_file():
            raise HTTPException(404, "Unknown batch")
        data = path.read_bytes()
        return Response(data, media_type="text/csv", headers={"X-Content-SHA256": hashlib.sha256(data).hexdigest()})
    return app


app = create_app()
