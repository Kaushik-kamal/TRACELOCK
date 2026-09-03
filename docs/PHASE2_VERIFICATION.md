# Phase 2 — Acquisition + Verification Firewall

> **DISCOVERY MAKES CLAIMS. TRACELOCK RE-MEASURES EVIDENCE.**

Status: **complete and live-validated against the real Phase 0 artifact.**
All success criteria met. One finding needs a decision before the demo — §Live results.

---

## Live results — read this first

Run against `search_gate_20260831T150728Z_0ceeeb0a.json`, all 20 real candidates:

```
  discovered by search engine : 20
  independently examined      : 20
  VERIFIED                    : 0
  INCONCLUSIVE                : 0
  REJECTED                    : 20

  rejection reasons:
    19x  LOW_FACE_SIMILARITY (at COMPARED)
     1x  HTTP_ERROR (at ACQUIRED)
```

**The verifier is proven. The candidate set contains no true positive.**

Google Lens returned 20 LinkedIn-style headshots that are *compositionally*
similar to the probe — same framing, same crop convention, similar subject
demographics — and are **different people**. Every one was re-downloaded,
re-decoded, re-detected, and re-measured, and every one was rejected with an
explicit machine-readable reason.

### The observed distribution supports the threshold

| n | min | max | mean | sd |
|---|---|---|---|---|
| 19 | −0.0199 | 0.2934 | 0.1434 | 0.0921 |

This is a textbook **impostor distribution** for ArcFace: centred near zero,
tight, topping out well below any plausible genuine-pair score. It is weak
evidence that the model and the provisional floor (0.35) are behaving sanely.

**It is NOT calibration.** There is not a single genuine pair in this set, so
no FAR/FRR can be computed from it. Phase 1's calibration architecture still
has no data, and this run does not change that.

### What this confirms about the discovery layer

The original architecture blueprint stated:

> Google Lens finds the *same photo republished*; it does not do face identity
> search. Yandex does genuine face matching.

This run confirms it empirically. Lens did not even return the probe's own
source page — it matched *style*, not *person*.

**The fix is at the discovery layer, not the verification layer**, and Phase 0
already has the machinery:

```bash
python scripts\search_viability_gate.py --image data\probes\me.jpg --image-url "https://..." --engine yandex_images
```

Then re-run Phase 2 against the new artifact. Nothing in Phase 2 needs to
change — which is the multi-provider abstraction paying for itself.

**This 20/20 rejection is a strong demo asset, not an embarrassment.** A naive
"trust the search engine" pipeline would have presented 20 strangers as
evidence. It is worth showing on camera *alongside* a run that produces a
survivor.

---

## Architecture

```
src/tracelock/acquisition/
    fetcher.py      secure download: SSRF guard, streaming size cap, redirects
    validation.py   magic bytes, real decode, decompression-bomb limits
    cas.py          content-addressed store, explicit deduplication
    provenance.py   eTLD+1 via the Public Suffix List (offline)

src/tracelock/verification/
    phash.py        DCT perceptual hash, definition pinned in-repo
    relation.py     the 2x2 evidence structure
    policy.py       provisional thresholds, structurally uncalibratable
    models.py       results, stage history, evidence records
    verifier.py     the DISCOVERED -> CLASSIFIED pipeline

src/tracelock/core/
    reasons.py      stages + rejection taxonomy (SHARED, see below)
```

### One structural change to an earlier phase

`reasons.py` was written into `verification/` and **moved to `core/`**.

`acquisition` raises rejection reasons; `verification` classifies on them.
Placing the taxonomy in either package created a genuine import cycle — which
is precisely the signal that it belongs to neither. `core/` depends on nothing
internal by design, so shared vocabulary lives there.

No Phase 0 or Phase 1 behaviour changed. All 314 of their tests still pass.

### Integration boundary — no Phase 1 changes needed

`FaceEngine.analyze(path)` already takes a path, and **CAS blobs are paths
holding the original bytes**. That preserves the required distinction for free:

```
ORIGINAL EVIDENCE BYTES   ->  what the CAS stores and what gets hashed
MODEL INPUT REPRESENTATION -> what the engine decodes internally, discarded
```

Validation decodes to inspect and throws the array away. A test asserts the
input bytes are unchanged after validation.

---

## Part I — the Phase 1 carry-forward, fixed

Phase 1 flagged `core.models.registrable_host()` as wrong for multi-label
public suffixes. `acquisition/provenance.py` now supplies the correct version
via `tldextract`.

**Phase 0's helper was deliberately left alone.** It is a display-field
fallback whose name honestly says *host*, and its contract has not changed.
Mutating a shipped Phase 0 contract to fix a Phase 3 scoring concern would be
the wrong trade.

```
registrable_host()    core/models.py           display fallback, approximate
registrable_domain()  acquisition/provenance.py PSL-correct, for Phase 3
```

### A decision Phase 3 must settle

`tldextract` has a "private" PSL section covering hosting platforms:

| `include_private` | `user-a.github.io` / `user-b.github.io` | Effect |
|---|---|---|
| `False` (**default**) | both → `github.io` | **under**-counts publishers |
| `True` | two distinct domains | gameable: free subdomains are cheap |

Phase 3 multiplies trust by independent-domain count, so **over-counting
inflates a trust score while under-counting only makes us conservative**. The
default is the direction that cannot inflate trust. Flag is exposed; Phase 3
decides.

Offline is enforced (`suffix_list_urls=()`) so the suite never fetches a PSL
and results do not vary with when they were run.

---

## Part A — what the fetcher refuses to trust

| Control | Why |
|---|---|
| scheme allowlist | blocks `file://`, `data://`, `ftp://` |
| **SSRF guard** | provider URLs are untrusted; `169.254.169.254` and `127.0.0.1` are real targets |
| streaming size cap | enforced **per chunk**, not from `Content-Length` |
| redirect limit | bounded; also covers loops |
| magic bytes | authoritative — headers are recorded, never trusted |
| header allowlist | `Set-Cookie` / `Authorization` never reach the artifact |

**Why the size cap is enforced mid-stream:** a `Content-Length` pre-check is
trivially defeated — a server can omit it, understate it, or use chunked
encoding. A test proves this by having a mock server declare `content-length:
100` and then send 10 KB; the byte counter aborts it.

**Documented, not silently ignored:** the SSRF check is a check-then-use race
(DNS rebinding). Closing it needs IP pinning into the connection, which httpx
does not expose cleanly.

---

## Part E — the 2×2, kept independent

|  | LOW pHash sim | HIGH pHash sim |
|---|---|---|
| **HIGH face sim** | `SAME_PERSON_DIFFERENT_PHOTO` | `SAME_PHOTO_REPUBLISHED` |
| **LOW face sim** | `UNRELATED` | `VISUAL_MATCH_FACE_MISMATCH` |

Both raw signals are always emitted separately so a reader can **re-derive the
quadrant rather than trust it**.

`EvidenceRelation.is_independent_corroboration` is `True` for exactly one
quadrant — `SAME_PERSON_DIFFERENT_PHOTO`. A republished copy of the same image
is *provenance*, not corroboration, and Phase 3 must not double-count it as
independent evidence.

`VISUAL_MATCH_FACE_MISMATCH` is the anomaly quadrant: near-identical imagery
whose faces do not match means a crop, a shared template, a stock photo, or an
altered image. Surfaced, not discarded.

### pHash definition (pinned — changing it invalidates stored hashes)

Implemented in-repo rather than imported, for the same reason the ROC code is:
in a forensic system the definition of a measurement should be readable here.

Greyscale → 32×32 `INTER_AREA` → DCT-II → top-left 8×8 → **median excluding
the DC term** → 64 bits. The DC exclusion matters: it carries overall
brightness, so including it would make the hash track exposure rather than
structure. A test confirms a +28 brightness shift moves the hash ≤12 bits.

---

## Part D — the honest position on thresholds

We have no calibration data. The wrong response is a plausible-looking number
treated as a decision boundary. The right response is a **wide inconclusive
band**:

```
similarity < 0.35             REJECTED      clearly not supported
0.35 <= similarity < 0.55     INCONCLUSIVE  measured, cannot conclude
similarity >= 0.55            VERIFIED      supported, still provisional
```

The 0.20-wide gap **is our admitted uncertainty made visible.** Narrowing it
requires data; narrowing it without data would be manufacturing confidence.

### Structural guards, not conventions

```python
VerificationPolicy(calibrated=True)   # raises ValueError
```

`calibrated` cannot be set to `True`. Claiming calibration requires a fitted
mapping backed by labelled pairs, not an edit. The policy serializes into every
artifact carrying `thresholds_are_provisional: true` and an explicit disclaimer.

No percentage or probability of identity is emitted anywhere. A test walks
every string in the artifact asserting no `"% same"` / `"% match"` phrasing.

---

## Two bugs found during implementation

**1. `failed_stage` was `None` on a similarity rejection.** A
`LOW_FACE_SIMILARITY` rejection returns through the *success* path, where no
stage recorded `FAILED` — every stage genuinely ran fine; the candidate failed
on the measurement. So `failed_stage` was empty, breaking the phase's core
promise that every rejection names its stage. Fixed by deriving it from the
reason, which keeps `primary_reason.stage == failed_stage` true on **every**
rejection path. Regression test added.

**2. A Unicode URL crashed the renderer and destroyed the run.** Candidate #5's
source URL contains mathematical-bold characters (`𝗜 𝘄𝗮𝘀𝗻𝘁`). Windows'
cp1252 console raised `UnicodeEncodeError` *while printing*, aborting the run
and writing no artifact — turning a display problem into destroyed evidence.
Fixed by forcing UTF-8 on stdout plus a per-line fallback.

This one matters for the demo: a crash mid-recording on a real-world URL would
be unrecoverable on camera.

---

## Testing

**443 tests, all passing.** No test touches the live internet.

| Suite | Tests |
|---|---|
| Phase 0 (unchanged) | 92 |
| Phase 1 (unchanged) | 222 |
| **Phase 2 acquisition** | **74** |
| **Phase 2 verification** | **55** |

Pipeline tests use a fake face engine returning **real `Embedding` objects** —
deliberately not a duck-typed stand-in, because `cosine_similarity` accepts
only `Embedding` or a raw array, and that strictness is what stops incompatible
vector types being compared silently.

### The invariants that make rejection transparency real

- **N candidates in, N results out.** Six candidates failing six different
  ways still produce six results.
- **Every rejection names a reason AND a stage**, and they agree.
- **Stage history records SKIPPED explicitly** — the absence of a measurement
  is recorded, not inferred from a missing field.
- **No raw embedding reaches the artifact** — only a quantized SHA-256.
- **Exact and near duplicates stay distinct** — a test recompresses an image
  and asserts different `sha256` but near-identical pHash.

---

## Artifact

`data/runs/verify_<timestamp>_<probe-prefix>.json`, schema
`verification-run/1`, shaped for the Phase 4 notary.

Audited clean: no API keys, no raw embeddings, no cookies, no auth headers.
The probe appears only as `sha256` plus `embedding_quantized_sha256` — a
commitment, never the vector.

---

## Phase 2 success criteria

| Criterion | Status |
|---|---|
| Candidates loaded from a real Phase 0 artifact | ✅ 20, no hardcoded URLs |
| Media actually re-downloaded | ✅ 19/20; 1 genuine 404 |
| Downloaded bytes SHA-256 hashed | ✅ |
| Content stored through CAS | ✅ 19 blobs, 223 KB |
| Actual image bytes validated | ✅ magic bytes + real decode |
| Face analysis runs independently | ✅ re-detected on our bytes |
| Face similarity measured | ✅ |
| pHash similarity measured | ✅ |
| Exact and near duplicates distinguished | ✅ separate signals |
| Rejection reasons explicit | ✅ 21-reason taxonomy, stage-mapped |
| At least one live candidate rejected with evidence | ✅ **20** |
| Results persisted as an artifact | ✅ |
| Phase 0 + Phase 1 tests green | ✅ 314/314 |

---

## Carried into Phase 3

1. **Get a discovery set containing a true positive.** Re-run Phase 0 with
   `--engine yandex_images`. This is the highest-value next action and needs no
   new code.
2. **Calibration still has no data.** The impostor-only distribution above is a
   sanity signal, not a calibration. Genuine pairs are required before any
   threshold can stop being provisional.
3. **`include_private` PSL decision** for corroboration counting (above).
4. **Phase 3 must not treat `SAME_PHOTO_REPUBLISHED` as corroboration** —
   `is_independent_corroboration` already encodes this.
5. **SSRF DNS-rebinding race** remains open; documented in `fetcher.py`.
