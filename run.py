"""Launch TRACELOCK.

    python run.py

Serves the UI and the API from one process on http://127.0.0.1:8000

One process on purpose: the face engine is loaded once and reused across
requests, so it must outlive a single call. A separate frontend dev server
would also mean two commands and one more thing to fail on demo day.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import uvicorn  # noqa: E402

HOST = "127.0.0.1"
PORT = 8000


def main() -> int:
    print("=" * 70)
    print("  TRACELOCK  ·  Digital Identity Evidence & Provenance Engine")
    print("=" * 70)
    print("  UI      http://{0}:{1}".format(HOST, PORT))
    print("  API     http://{0}:{1}/docs".format(HOST, PORT))
    print()
    print("  The face model warms in the background (~7s); the UI is usable")
    print("  immediately. Blockchain credentials are optional - without them")
    print("  TRACELOCK uses its local demo chain.")
    print("  Ctrl+C to stop.")
    print("=" * 70)

    uvicorn.run("tracelock.api:app", host=HOST, port=PORT, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
