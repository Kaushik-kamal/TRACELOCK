"""FastAPI application exposing the TRACELOCK pipeline."""

from tracelock.api.app import app, create_app

__all__ = ["app", "create_app"]
