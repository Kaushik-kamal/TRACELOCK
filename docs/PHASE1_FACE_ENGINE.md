# Phase 1 — Face Intelligence Engine

Status: **complete and live-validated.** The engine produces measurements. It
makes no identity decisions; that requires calibration and lands in Phase 3.

---

## Environment smoke test (the Phase 1 gate)

Run before anything else. The Phase 0 checklist flagged `insightface` on Python
3.12 as an open risk.

**Result: the risk did not materialise.**

| | |
|---|---|
| `insightface` | **1.0.1**, installed as a pure-Python wheel — **no compiler required** |
| `onnxruntime` | 1.29.0 (`cp312` wheel), `CPUExecutionProvider` |
| `numpy` / `opencv` | 2.5.2 / 5.0.0 |
| Python | 3.12.6 |
| Model download | ~275 MB to `~/.insightface/models/buffalo_l`, one time |
| Cold init | ~48 s including download; ~2 s after |
| Warm inference | ~2.1 s per image |

> Do **not** downgrade to `insightface` 0.7.x. That version is sdist-only and
> needs MSVC Build Tools to compile a Cython extension. 1.0.1 avoids it entirely.

### Correction to the architecture blueprint

The blueprint stated buffalo_l uses **ArcFace R100**. It does not.

```
det_10g.onnx      SCRFD-10GF          detection
w600k_r50.onnx    ArcFace ResNet50    recognition, 112x112 in, 512-d out
2d106det.onnx     2D landmarks (106)
1k3d68.onnx       3D landmarks (68)   -> pose
genderage.onnx    attribute estimates
```

buffalo_l's recognition model is **ResNet50** trained on WebFace600K. R100 is
the `antelopev2` pack. Provenance reporting has to name the model that actually
ran, so this is recorded accurately in `ModelProvenance.recognition_model`.

---

## Measured facts

Everything below was measured on this build, not assumed.

### Determinism — stronger than expected

| Condition | Result |
|---|---|
| Repeated calls, same process | **bit-identical** (max abs diff 0.0) |
| Separate interpreter processes | **bit-identical**, including raw float32 bytes |
| `OMP_NUM_THREADS=1` vs default | **bit-identical** |
| `det_score` across all runs | identical to 10 decimal places |

This **revises the blueprint's risk ranking.** Float non-determinism was listed
as high-likelihood. On this build it does not occur at all.

Quantization is retained anyway, because the actual Phase 4 threat is a
*verifier on a different machine* — different ONNX Runtime build, different
execution provider, different hardware. `verify_determinism()` re-checks at
runtime rather than trusting this measurement to hold forever.

### Quantization round-trip

`int8_scaled` at scale 127:

- round-trip cosine against the original: **0.998635**
- occupies only `[-24, 15]` of the int8 range

The range under-use is deliberate and harmless. The output is a **commitment
input, never a reconstruction target**, and coarser quantization is *more*
robust to cross-machine drift — which is the whole point.

### Pose units — resolved empirically

`face.pose` returns `(pitch, yaw, roll)`. Whether those were degrees or radians
mattered for the frontality metric, so it was tested by rotating the input:

| Applied rotation | pitch | yaw | roll |
|---|---|---|---|
| 0° | −0.046 | 0.526 | 0.079 |
| +10° | 0.079 | 0.457 | **−9.814** |
| +20° | 0.235 | 0.073 | **−19.764** |
| −15° | 0.477 | 0.157 | **+15.304** |

**Degrees.** Roll tracks the applied rotation one-for-one while yaw and pitch
stay flat — which also confirms that roll is purely in-plane and is therefore
**excluded from the frontality metric**: the 5-point similarity alignment
already corrects it, and penalising it would double-count a solved problem.

### Blur sensitivity study (n = 1)

Progressive Gaussian blur on the live probe, measuring face-crop Laplacian
variance against cosine similarity to the unblurred embedding:

| kernel | lap_var | cos to original |
|---|---|---|
| 1 (none) | 104.2 | 0.9998 |
| 5 | 28.7 | 0.9968 |
| 9 | 13.2 | 0.9911 |
| 17 | 5.6 | 0.9676 |
| 21 | 4.4 | 0.9472 |
| 27 | 3.4 | 0.8955 |
| 35 | 2.8 | 0.8200 |

**This changed the code.** `SHARPNESS_REFERENCE` was initially guessed at
`500.0`, which scored a perfectly usable face at 0.21 and raised a false
`BLURRY_FACE` warning. It is now `100.0`, anchored to the measurement above.

**What this study does NOT establish.** It is one subject, one image, synthetic
Gaussian blur. More importantly, a *self-similarity* sweep cannot detect the
failure mode this metric exists to catch: blur pulls embeddings toward the
population mean and **inflates impostor similarity**. Detecting that needs
labelled impostor pairs, which Phase 1 does not have. The metric therefore
stays flagged `uncalibrated=True`.

---

## Architecture

```
src/tracelock/face/
    errors.py       error taxonomy -- pipeline control flow, not diagnostics
    models.py       frozen dataclasses; immutable embeddings
    selection.py    primary-face policy (pure, model-free, fast to test)
    quality.py      six interpretable metrics + geometric aggregation
    similarity.py   cosine / angular utilities. NO thresholds.
    engine.py       FaceEngine.analyze() -> FaceAnalysisResult

src/tracelock/calibration/
    contract.py     dataset contract + the anti-fabrication mechanism
    metrics.py      ROC / FAR / FRR / EER / AUC / Wilson intervals
    loaders.py      ManifestLoader (real, consented) | FixtureLoader (synthetic)
```

### Deviations from the Phase 1 specification

Three, each deliberate:

**1. `selection.py` split out of `engine.py`.** The primary-face policy is a
documented decision procedure that must be testable without loading a 275 MB
model. All 30 selection tests run on plain numbers in milliseconds.

**2. `calibration/` is a sibling package, not `face/calibration.py`.** Phase 3's
trust scorer needs calibration but must not import the face engine to get it.
Keeping them separate prevents a scoring → face dependency.

**3. Frozen dataclasses inside `face/`, not Pydantic.** Phase 0 used Pydantic
for `Candidate` because that parses untrusted third-party JSON. Face results are
the opposite problem: they originate in-process from numpy and need read-only
enforcement on a 512-float array. Serialization is explicit via `to_dict()`.
Phase 0's Pydantic models are untouched.

---

## Primary-face selection policy

```
score = 0.50 * area_rel + 0.30 * det_score + 0.20 * centrality
```

- **`area_rel`** — face area ÷ *largest face area in this image*. Relative, not
  absolute: selection is a comparison among the faces present, and absolute area
  ratios (a face is often 5–15% of a frame) compress into a range too narrow to
  discriminate.
- **`det_score`** — the detector's own confidence. The guard rail that stops a
  spurious low-confidence blob winning on size alone.
- **`centrality`** — normalized by the half-diagonal, so it is
  resolution-independent. Weighted lowest because framing convention is real but
  subjects are not reliably centred.

**Ambiguity is reported, never hidden.** The top-1 vs top-2 margin is always
computed. Below `ambiguity_threshold` (default 0.10) the engine attaches an
`AMBIGUOUS_PRIMARY_FACE` warning; `strict_ambiguity=True` raises instead.
Returning a flagged best guess beats both silent false certainty and refusing to
produce any result.

**Determinism** comes from a total order: `(−score, −area, x1, y1)`. Two faces
can only remain tied if they occupy the identical rectangle. Tested against
input reordering and repeated invocation.

---

## Quality metrics

Every metric states what it measures, why it matters, and **how it fails**.
A metric whose failure mode you cannot state is one you cannot defend.

| Metric | Weight | Anchor | Key failure mode |
|---|---|---|---|
| `face_pixel_size` | 0.30 | **112px** = model input | Says nothing about whether those pixels are sharp |
| `sharpness` | 0.25 | n=1 blur study | Cannot detect impostor-similarity inflation |
| `detection_confidence` | 0.20 | native | Detects *faces*, including posters and statues |
| `pose_frontality` | 0.15 | **60°** degradation | Pose estimate is least reliable at extreme pose — degrades exactly where it matters |
| `exposure_integrity` | 0.05 | working default | Small specular highlights count as clipping |
| `frame_containment` | 0.05 | working default | Detects frame truncation only, **not occlusion** |

Two anchors are principled (112px is the recognition model's literal input size;
60° is where ArcFace degrades sharply). The rest are working defaults and are
flagged `uncalibrated=True` in every output.

### Aggregation: weighted **geometric** mean

```
Q = exp( Σ wᵢ · ln(max(scoreᵢ, ε)) )
```

Geometric, not arithmetic, for the same reason the Phase 3 trust score is
multiplicative: **a necessary condition must gate, not merely contribute.** A
20px face is unusable no matter how sharp and well-lit it is, and an arithmetic
mean would let the strong terms average that away. A test asserts the geometric
aggregate is strictly lower than the arithmetic one when any metric collapses.

### A fairness decision worth stating

`exposure_integrity` measures **clipping** (pixels at 0 or 255), not mean
brightness. Thresholding on mean luminance systematically penalises darker skin
tones and is a well-documented source of demographic bias in face systems.
Clipping is tone-neutral — a saturated pixel is destroyed information regardless
of the subject. Mean luminance is still *reported* as a diagnostic, but it is
deliberately **not scored**. Two tests assert that uniformly dark and uniformly
bright but unclipped patches are not penalised.

---

## What Phase 1 refuses to do

There is **no identity threshold anywhere in the face package**, and a test
enforces it by scanning the source for `IDENTITY_THRESHOLD`, `SAME_PERSON`,
`MATCH_THRESHOLD`, and `is_same_person`. A second test asserts `similarity.py`
contains no `-> bool` return.

A cosine similarity is a geometric quantity. An identity claim is a statistical
one. Writing `if sim > 0.5` would fabricate a calibrated result that does not
exist.

---

## Calibration: architecture without empirical claims

Phase 1 ships the **contract**, the **loaders**, and the **metrics**. It ships
**no real calibration data and claims no results.**

### The anti-fabrication mechanism

The temptation in a competition is to generate plausible pairs, fit a ROC, and
present it as evidence. That is the most serious integrity failure available —
worse than shipping with no calibration at all.

So every dataset declares its kind, and the stamp **propagates structurally**:

```
DatasetKind.FIXTURE  ->  ObservationSet.is_fixture  ->  CalibrationReport.is_fixture
                                                              |
                                                              v
                                              render() prints a refusal banner
                                              and WITHHOLDS the metrics
```

`citable` requires `kind == REAL` **and** at least one obtained consent record.
`ManifestLoader` refuses a manifest with no consent unless `require_consent=False`
is passed explicitly for a licence-verified public dataset.

Eight tests in `TestAntiFabrication` enforce this, including that a fully
consented *fixture* is still non-citable — consent cannot launder synthetic data.

### What the fixture is legitimately for

Verifying the maths. `FixtureLoader` generates unit vectors with controllable
angular separation, which confirms ROC/EER/FAR/Wilson code is correct. It says
nothing about face recognition performance, and the type system makes that
inescapable.

### Toward real calibration

1. Assemble ~30 genuine and ~200 impostor pairs of **consented** subjects.
2. Write `data/calibration/manifest.json` with consent records.
3. `observe_dataset(dataset, engine)` → `ObservationSet`.
4. `evaluate(observations)` → `CalibrationReport` with EER, AUC, and operating
   points at FAR = 1% and 0.1%.
5. Report the **Wilson interval**. At n ≈ 30 it is wide, and saying so is what
   separates a measurement from a decoration.

**FAR is the number that matters here**, not EER. A false accept means asserting
an innocent person appears in discovered content; a false reject only means a
missed candidate. The costs are not symmetric, so `threshold_at_far()` is the
operating point to use.

---

## Live validation

```powershell
python scripts\face_engine_report.py --image data\probes\me.jpg --verify-determinism
```

Result on the real probe:

```
faces detected  : 1
bounding box    : 216x266 px
det confidence  : 0.8753
pose            : pitch=-0.05  yaw=+0.53  roll=+0.08   (0.53 deg out-of-plane)
embedding       : 512-d, L2 norm 1.0000000000
quality         : 0.6570 [GOOD]
determinism     : bit identical, max abs diff 0.000e+00
```

The 512-d vector is **never printed** — it is biometric data. The report shows
dimension, norm, and a truncated digest of the quantized form so runs can be
compared without exposing it.

---

## Test fixture and consent policy

**No biometric data is committed to this repository.** `data/probes/` is
gitignored.

Tests that need a real face use `data/probes/me.jpg`, supplied by the operator
under the consent policy in the README. They are marked `needs_model` /
`needs_probe` and **skip cleanly** when either is absent, so a fresh clone still
runs green.

Everything testable without a face — dataclass invariants, immutability,
quantization, selection policy, quality maths, calibration metrics — runs
always, on synthetic patches and plain numbers.

---

## Phase 1 success criteria

| Criterion | Status |
|---|---|
| InsightFace initializes in this environment | ✅ 1.0.1, pure-python wheel |
| buffalo_l runs real inference | ✅ 5 sub-models loaded |
| Probe face detected | ✅ 1 face, det 0.8753 |
| 512-d embedding extracted | ✅ |
| Embedding normalization verified | ✅ L2 = 1.0000000000 |
| Quality metrics computed | ✅ 6 metrics + geometric aggregate |
| Primary-face policy deterministic | ✅ total order, tested against reordering |
| Similarity utilities tested | ✅ incl. zero-vector and dim-mismatch |
| Error taxonomy exists | ✅ 12 types across 4 families |
| Calibration architecture, no fake claims | ✅ structural fixture stamping |
| Real probe validation succeeds | ✅ |
| Phase 0 tests still pass | ✅ untouched |
| No hardcoded identity decisions | ✅ enforced by test |

---

## Open items carried into Phase 2

1. **Impostor-pair blur study.** The n=1 sweep cannot detect similarity
   inflation under blur. Needs labelled impostor pairs.
2. **`registrable_host()` still approximates eTLD+1** by stripping `www.`
   (Phase 0 known limitation). Must become `tldextract` before corroboration
   counts independent domains.
3. **Sharpness normalization is resolution-dependent.** A constant reference
   across wildly different image sizes is a simplification worth revisiting.
4. **Minors safeguard not yet enforced.** `age_estimate` is surfaced but the
   engine does not gate on it — that is a pipeline policy decision, not an
   engine one.
