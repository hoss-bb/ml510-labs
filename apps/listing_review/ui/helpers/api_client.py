"""HTTP access without Streamlit or application-runtime dependencies."""
import httpx


class ServiceError(Exception):
    def __init__(self, message, request_id=None):
        super().__init__(message)
        self.request_id = request_id


class ApiClient:
    def __init__(self, base_url):
        self.base_url = base_url

    def request(self, method, path, **kwargs):
        try:
            with httpx.Client(base_url=self.base_url, timeout=15) as client:
                response = client.request(method, path, **kwargs)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as error:
            messages = {
                404: "This record is no longer available. Refresh the listings.",
                422: "Check the entered fields and try again.",
                409: "This request conflicts with an existing request. Refresh before trying again.",
                503: "The service is not ready. Your saved listings and reviews have not been changed.",
            }
            raise ServiceError(messages.get(error.response.status_code,
                "The service could not complete this request. Please try again."),
                error.response.headers.get("X-Request-ID")) from error
        except httpx.RequestError as error:
            raise ServiceError("Cannot reach the application service. Please try again shortly.") from error
        except ValueError as error:
            raise ServiceError("The service returned an invalid response. Please try again.") from error
