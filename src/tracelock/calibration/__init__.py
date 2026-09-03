"""Calibration: converting similarity scores into defensible identity claims.

Phase 1 ships the CONTRACT, LOADERS and METRICS. It ships no real calibration
data and claims no empirical results. Anything derived from a FIXTURE dataset
is stamped and refuses to render as evidence -- see `contract.py`.
"""

from tracelock.calibration.contract import (
    CalibrationDataset,
    CalibrationPair,
    ConsentRecord,
    DatasetKind,
    DatasetProvenance,
    ObservationSet,
    PairLabel,
    SimilarityObservation,
)
from tracelock.calibration.loaders import (
    CalibrationLoader,
    FixtureLoader,
    ManifestLoader,
    observe_dataset,
)
from tracelock.calibration.metrics import (
    FIXTURE_BANNER,
    CalibrationReport,
    OperatingPoint,
    RocCurve,
    equal_error_rate,
    evaluate,
    roc_curve,
    threshold_at_far,
    wilson_interval,
)

__all__ = [
    "PairLabel",
    "DatasetKind",
    "ConsentRecord",
    "CalibrationPair",
    "DatasetProvenance",
    "CalibrationDataset",
    "SimilarityObservation",
    "ObservationSet",
    "CalibrationLoader",
    "ManifestLoader",
    "FixtureLoader",
    "observe_dataset",
    "RocCurve",
    "OperatingPoint",
    "CalibrationReport",
    "roc_curve",
    "equal_error_rate",
    "threshold_at_far",
    "wilson_interval",
    "evaluate",
    "FIXTURE_BANNER",
]
