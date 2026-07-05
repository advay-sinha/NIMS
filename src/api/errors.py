"""API error types.

Purpose
-------
Typed errors carrying an HTTP status code, raised by the loader/service and
converted to clean JSON responses by the app's exception handlers. Stack
traces stay in the server logs — they are never exposed to clients.
"""

from __future__ import annotations


class ApiError(RuntimeError):
    """Base API error; subclasses set the HTTP status code."""

    status_code = 500


class ModelNotAvailableError(ApiError):
    """No servable model for the requested dataset/stage (or files missing)."""

    status_code = 404


class RequestValidationError(ApiError):
    """The uploaded rows cannot be transformed into the model's features."""

    status_code = 422


class PayloadTooLargeError(ApiError):
    """The request exceeds the configured row limit."""

    status_code = 413


class InferenceFailureError(ApiError):
    """The transform or prediction failed server-side (details in logs)."""

    status_code = 500
