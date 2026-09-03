"""Structured, readable stage failures.

A stage that times out, is blocked, or is refused must produce a RESULT, not a
traceback. The operator sees a sentence they can act on; the artifact keeps the
machine-readable stage and status so a later reader knows exactly where the
pipeline stopped and why.

Two rules govern everything here:

  * A failure never carries findings. There is no partial trust score, no
    "probable" candidate list, nothing that could be mistaken for evidence.
  * A failure says what could not be DONE, never what does not EXIST. "Could
    not be retrieved" and "does not exist" are different claims, and only the
    first one is ours to make.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FailureStatus(str, Enum):
    TIMEOUT = "timeout"
    BLOCKED = "blocked"
    UNAVAILABLE = "unavailable"
    REFUSED = "refused"
    INVALID_INPUT = "invalid_input"
    NOT_CONFIGURED = "not_configured"


# Per-stage timeouts, in seconds. No stage may hang indefinitely; each one is
# bounded by whichever of these applies to it.
STAGE_TIMEOUTS: dict[str, float] = {
    "url_resolution": 15.0,
    "image_download": 20.0,
    "search": 25.0,
    "candidate_fetch": 20.0,
    "face_analysis": 30.0,
    "anchoring": 60.0,
}


@dataclass(frozen=True, slots=True)
class StageFailure:
    """One stage's failure, in a shape both a person and a parser can read."""

    stage: str
    status: FailureStatus
    message: str
    recovery: tuple[str, ...] = field(default_factory=tuple)
    detail: str = ""
    timeout_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status.value,
            "stage": self.stage,
            "message": self.message,
        }
        if self.recovery:
            payload["recovery"] = list(self.recovery)
        if self.detail:
            payload["detail"] = self.detail
        if self.timeout_seconds is not None:
            payload["timeout_seconds"] = self.timeout_seconds
        return payload


# The three things an operator can always do when a public source will not
# yield an image. Local analysis is deliberately included: it is the honest
# fallback that never uploads anything, and leaving it out is what turns a
# blocked platform into a dead end.
RECOVERY_UPLOAD = "upload_image"
RECOVERY_DIRECT_URL = "use_direct_url"
RECOVERY_LOCAL = "continue_local"

DEFAULT_RECOVERY = (RECOVERY_UPLOAD, RECOVERY_DIRECT_URL, RECOVERY_LOCAL)

RECOVERY_LABELS: dict[str, str] = {
    RECOVERY_UPLOAD: "Upload the image directly",
    RECOVERY_DIRECT_URL: "Use a direct public image URL",
    RECOVERY_LOCAL: "Continue with Local Analysis",
}


def timeout(stage: str, *, seconds: float | None = None, what: str = "") -> StageFailure:
    """A stage exceeded its budget. Never a crash, never a hang."""
    limit = seconds if seconds is not None else STAGE_TIMEOUTS.get(stage)
    subject = what or stage.replace("_", " ")
    return StageFailure(
        stage=stage,
        status=FailureStatus.TIMEOUT,
        message="{0} did not respond in time.".format(subject[:1].upper() + subject[1:]),
        detail=(
            "The step was stopped after {0:.0f} seconds so the investigation "
            "could not hang.".format(limit)
            if limit
            else "The step was stopped so the investigation could not hang."
        ),
        recovery=DEFAULT_RECOVERY,
        timeout_seconds=limit,
    )


def platform_blocked(platform_display: str, note: str, guidance: str = "") -> StageFailure:
    """A platform declined to serve the image to an anonymous request.

    Phrased as a retrieval failure, because that is what it is. Whether the
    image exists is not something a refusal tells us.
    """
    return StageFailure(
        stage="url_resolution",
        status=FailureStatus.BLOCKED,
        message=(
            "{0} post detected, but TRACELOCK could not retrieve the image "
            "publicly from this link.".format(platform_display)
        ),
        detail=" ".join(part for part in (note, guidance) if part),
        recovery=DEFAULT_RECOVERY,
    )


def unavailable(stage: str, message: str, detail: str = "") -> StageFailure:
    return StageFailure(
        stage=stage,
        status=FailureStatus.UNAVAILABLE,
        message=message,
        detail=detail,
        recovery=DEFAULT_RECOVERY,
    )


def blocked_target(stage: str, detail: str = "") -> StageFailure:
    """An SSRF refusal. Offers no 'try again' recovery, because it should not
    be retried -- the target is not one this system may reach."""
    return StageFailure(
        stage=stage,
        status=FailureStatus.REFUSED,
        message="That link points to a private or internal network address and was blocked.",
        detail=detail,
        recovery=(RECOVERY_UPLOAD, RECOVERY_LOCAL),
    )


def recovery_options(failure: StageFailure | None) -> list[dict[str, str]]:
    """Recovery actions as UI-ready rows. Always at least one way forward."""
    keys = failure.recovery if failure and failure.recovery else DEFAULT_RECOVERY
    return [{"action": key, "label": RECOVERY_LABELS[key]} for key in keys]
