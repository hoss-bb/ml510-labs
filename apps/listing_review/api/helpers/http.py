"""Register request logging and sanitized HTTP validation responses."""
from time import perf_counter
import uuid
import sqlite3
from starlette.exceptions import HTTPException
from ..schemas import ErrorResponse, ValidationIssue
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from lab_helpers.s10_shared.runtime.logging import event


def register_http_handlers(app):
    @app.middleware("http")
    async def timing(request: Request, call_next):
        request.state.request_id = uuid.uuid4().hex
        started = perf_counter()
        try:
            response = await call_next(request)
        except Exception as error:
            event("request", request_id=request.state.request_id, outcome="failed",
                  error_type=type(error).__name__)
            raise
        response.headers["X-Request-ID"] = request.state.request_id
        event("request", request_id=request.state.request_id, method=request.method,
              status=response.status_code, seconds=perf_counter() - started)
        return response

    def error_response(request, status, code, message, details=None, headers=None):
        reference = request.state.request_id
        body = ErrorResponse(code=code, message=message, request_id=reference, details=details or [])
        return JSONResponse(status_code=status, content=body.model_dump(),
                            headers={**(headers or {}), "X-Request-ID": reference})

    @app.exception_handler(HTTPException)
    async def http_error(request, error):
        codes = {404: "not_found", 405: "method_not_allowed", 409: "conflict", 503: "unavailable"}
        return error_response(request, error.status_code, codes.get(error.status_code, "http_error"),
                              str(error.detail), headers=error.headers)

    @app.exception_handler(RequestValidationError)
    async def invalid(request, error):
        # Never echo submitted values or exception context.
        issues = [ValidationIssue(field=".".join(map(str, item["loc"])), type=item["type"])
                  for item in error.errors()]
        return error_response(request, 422, "validation_error", "Check the indicated fields.", issues)

    @app.exception_handler(sqlite3.OperationalError)
    async def storage_unavailable(request, error):
        event("storage", request_id=request.state.request_id, outcome="failed", error_type=type(error).__name__)
        return error_response(request, 503, "unavailable", "Storage unavailable. Please try again shortly.")

    @app.exception_handler(Exception)
    async def unexpected(request, error):
        # Middleware logs the type. Keep internal messages out of the HTTP body.
        return error_response(request, 500, "internal_error", "The service could not complete this request.")
