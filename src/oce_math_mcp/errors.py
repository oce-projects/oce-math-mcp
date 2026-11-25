"""Expected tool errors: they become the `error` field of the response."""

from __future__ import annotations

CODES = frozenset(
    {
        "INVALID_ARGUMENT",
        "PARSE_ERROR",
        "FORBIDDEN",
        "LIMIT_EXCEEDED",
        "TIMEOUT",
        "MATH_ERROR",
        "NOT_IMPLEMENTED",
        "UNAVAILABLE",
        "INTERNAL_ERROR",
    }
)


class ToolError(Exception):
    """Error with code, message and optional corrected example."""

    def __init__(self, code: str, message: str, example: str | None = None) -> None:
        if code not in CODES:  # pragma: no cover - programming error
            raise ValueError(f"unknown error code: {code}")
        super().__init__(message)
        self.code = code
        self.message = message
        self.example = example
