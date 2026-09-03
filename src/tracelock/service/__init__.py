"""Service layer -- normalizes every input source and orchestrates the pipeline.

Imports the existing modules directly; it never shells out to the CLI scripts,
which remain the reproducible reference path over the same code.
"""

from tracelock.service.inputs import (
    InputError,
    InputKind,
    SourceType,
    TraceInput,
    from_bytes,
    from_google_drive,
    from_upload,
    from_url,
    from_webcam,
)

__all__ = [
    "TraceInput", "SourceType", "InputKind", "InputError",
    "from_bytes", "from_upload", "from_webcam", "from_url", "from_google_drive",
]
