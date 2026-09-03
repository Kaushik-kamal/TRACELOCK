# Verification pipeline

What happens to one candidate between "a search engine mentioned it" and
"it is evidence" — or, far more often, "it was rejected, and here is why".

---

## The premise

Everything a search provider returns is **unverified by construction**. A
provider claiming a relationship between two images is a ranking signal, not
evidence. The pipeline below exists to turn that signal into a measurement, or
to discard it.

---

## Stages

A candidate advances through six stages. It can fail out at any one, and where
it failed is recorded.

| Stage | What happens | Typical failures |
|---|---|---|
| `DISCOVERED` | Returned by Lens or Yandex, deduplicated, ranked | `NO_MEDIA_URL`, `INVALID_URL` |
| `ACQUIRED` | Downloaded through the SSRF guard into the CAS | `HTTP_ERROR`, `DOWNLOAD_TIMEOUT`, `BLOCKED_URL_TARGET`, `CONTENT_TOO_LARGE` |
| `VALIDATED` | Decoded and checked as a real image | `NOT_AN_IMAGE`, `CORRUPT_IMAGE`, `IMAGE_TOO_SMALL`, `DECOMPRESSION_BOMB` |
| `ANALYZED` | Face detected, quality gated, embedded | `NO_FACE_DETECTED`, `LOW_FACE_QUALITY`, `AMBIGUOUS_MULTIPLE_FACES` |
| `COMPARED` | Cosine similarity against the probe embedding | `EMBEDDING_INCOMPATIBLE` |
| `CLASSIFIED` | Banded into a status | `LOW_FACE_SIMILARITY`, `DUPLICATE_CONTENT` |

The full vocabulary is 21 rejection reasons in
`src/tracelock/core/reasons.py`. Each carries the stage it belongs to and a
human explanation, and the UI renders that explanation verbatim rather than
inventing its own wording.

### Why the distinction between stages matters

A candidate rejected at `ACQUIRED` was **never compared to anything**. Showing
it in a gallery of "rejected candidates" alongside a similarity score would
imply a measurement that never happened. These are reported as a *not
retrievable* count instead.

Only candidates that reached `COMPARED` — that is, those with a real
`face_similarity` — are ever shown as scored cards.

---

## The decision

### Calibration

Raw cosine similarity is not a probability. A fitted Platt model maps it to
one:

```
P(same identity | s) = sigmoid(16.5779 · s − 5.9037)
```

Fitted from measured pairs and stored in `data/calibration/model.json`
alongside its own provenance and confidence note. `scripts/calibrate.py`
refits it; it spends no API credits and downloads nothing, because every
similarity involved was already measured during a real verification run.

### Bands

Two thresholds fall out of the model:

```
        LOW              INDETERMINATE              HIGH
  ───────────────┬──────────────────────────┬───────────────
              0.2938                     0.3528
   REJECTED         INCONCLUSIVE            VERIFIED_CANDIDATE
```

| Band | Status | What it means |
|---|---|---|
| `HIGH` | `VERIFIED_CANDIDATE` | Same person, on this evidence |
| `INDETERMINATE` | `INCONCLUSIVE` | Measured, insufficient to claim identity |
| `LOW` | `REJECTED` (`LOW_FACE_SIMILARITY`) | Different person |

The floor is the equal-error-rate threshold (0.293756). The ceiling adds the
inconclusive band width.

The middle band is the part worth defending. Collapsing it into a binary would
be easier to explain and would be a lie — a similarity of 0.32 genuinely does
not settle whether two photographs are the same person, and the honest answer
is to say so.

### The calibration is provisional, and says so

Fitted on **5 genuine and 35 impostor pairs**. AUC 0.9886, EER 0.0 — but the
95% confidence interval on that EER is `[0.0000, 0.4345]`. With a sample this
small the interval is wide, and the operating points are indicative rather
than definitive.

That sentence is stored in the model file itself under `confidence_note`, and
`thresholds_are_provisional` is carried through the verification policy into
the evidence bundle. Anyone reading an artifact can see the caveat without
reading this document.

---

## Terminal states

### `complete`

At least one candidate verified. Evidence is aggregated, a trust score is
produced, and the bundle is fingerprinted and anchored.

### `completed_no_match`

Every candidate was examined and none met the threshold. **This is a success.**

It was originally reported as `failed`, which was wrong: a run that discovered
44 candidates, downloaded 23, detected 23 faces and compared 23 embeddings did
its job perfectly and correctly declined to claim a match. Reporting that as a
failure taught the operator to distrust the single most valuable thing the
tool does.

`score_evidence` distinguishes the two refusals it previously conflated:

- `NoVerifiedEvidence` — everything was examined, nothing qualified. Terminal
  state `completed_no_match`.
- `UncalibratedScoreRefused` — there is no calibration model. A genuine
  failure, still reported as one.

In `completed_no_match` the UI shows the funnel, the highest similarity
observed, the threshold required, and the rejected candidates. It shows **no
trust score and no anchor**, and states that both require verified evidence
because producing either would be fabrication.

### `failed`

Something actually broke — no calibration model, discovery unavailable, probe
unreadable. The failure box names a recovery option.

---

## Evidence aggregation

Verified candidates are grouped by **independent registrable domain**. Ten
reposts of one image across one site is one corroboration, not ten.
Near-duplicate images are detected by perceptual hash (DCT pHash, Hamming
distance ≤ 10) and reported explicitly rather than silently merged.

The funnel is carried through to the artifact:

```
discovered → downloaded → validated → analysed → verified
                                    ↳ inconclusive
                                    ↳ rejected
                                    ↳ duplicates
```

Trust scoring weighs the number of independent publishers, metadata
completeness, and acquisition integrity.

---

## Fingerprinting and anchoring

The artifact is canonicalised with RFC 8785 JCS so the bytes are stable, then
hashed into 11 domain-separated Merkle leaves:

```
tl:schema              tl:run_id             tl:probe_commitment
tl:probe_model         tl:evidence_items     tl:evidence_funnel
tl:independent_domains tl:trust_score        tl:verification_policy
tl:source_artifacts    tl:pipeline
```

Domain separation means a value cannot be moved between leaves without
changing the root. Leaves are paired in sorted order when building the tree,
so the root does not depend on iteration order.

`verify_anchor.py` recomputes every leaf from the artifact on disk and, on a
mismatch, names **which leaf** changed. The UI's tamper test does the same
thing live: mutate one field and watch a specific leaf fail.

Only the root and packed metadata go on chain. `assert_no_biometric_leak`
raises if an embedding-shaped array is anywhere in the payload.

---

## What the pipeline will not do

- It will not promote an inconclusive candidate to verified.
- It will not produce a trust score without verified evidence.
- It will not anchor an empty result.
- It will not present a candidate it failed to download as a scored rejection.
- It will not describe a local demo chain as a public one.

Each of these is enforced by tests, not just convention.
