"""Safe, actionable review failures; never expose provider response bodies."""
from ai_lib.models.errors import ExternalCallError

from .messages import REVIEW_ERRORS


class ReviewError(ExternalCallError):
    def __init__(self, code):
        self.code = code
        super().__init__(REVIEW_ERRORS[code])
