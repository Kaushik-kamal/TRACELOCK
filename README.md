# TRACELOCK

Digital identity evidence and provenance engine. Give it a photograph of a
face; it searches the public web for that face, independently verifies every
candidate it finds, and produces a tamper-evident evidence bundle for the ones
that survive.

Built for HackHazards Goa 2026.

---

## The problem

Reverse image search is easy to call and easy to misread. Google Lens will
happily return two dozen photographs that "look like" the person you searched
for. Some are the same person. Most are strangers with a similar face shape,
similar lighting, or the same haircut.

If you build a tool that reports those hits as findings, you have built
something that confidently accuses innocent people. The search engine never
claimed those results were the same human being — it claimed they were
*visually similar*. That distinction is the entire problem.

It matters because the people who need this kind of tool are usually looking
for something painful: a stolen profile photo, an impersonation account, a
fake dating profile, a picture reposted somewhere it should not be. Being told
"we found 23 matches" when 22 of them are strangers is worse than being told
nothing.

## Why TRACELOCK exists

**Discovery is a commodity. Verification is the product.**

TRACELOCK treats every search result as an unverified claim. It re-downloads
each candidate image, runs its own face detection and embedding on the actual
pixels, and compares that against the probe using a calibrated threshold. Most
candidates get rejected, with a stated reason and a recorded score.

The rejections are not a failure mode — they are the demonstration. A system
that discards 22 of 25 hits and explains why is visibly reasoning. A system
that returns one perfect match is indistinguishable from a hardcoded script.

Two consequences shape the whole design:

- **A completed search that verifies nothing is a success**, not an error. It
  gets its own terminal state (`completed_no_match`) with the full funnel and
  the rejected candidates on screen.
- **No trust score and no blockchain anchor are produced without verified
  evidence.** Both require something real to attest to; generating either from
  an empty result would be fabrication.

---

## Core features

- **Universal input** — file upload, webcam capture, public image URL, social
  post link, or Google Drive link. Every path converges on the same pipeline.
- **Genuine reverse-image discovery** via SerpAPI (Google Lens + Yandex
  Images, queried concurrently). No hardcoded results, no fixtures.
- **Independent face verification** — SCRFD detection and ArcFace embeddings
  (InsightFace `buffalo_l`), run locally on the downloaded bytes.
- **Calibrated decisions** — a fitted Platt model converts cosine similarity
  into a probability, with an explicit inconclusive band between the floor and
  the ceiling.
- **Candidate evidence gallery** — verified, inconclusive, and rejected
  candidates shown with thumbnails, provenance, scores, and the pipeline's own
  rejection reason.
- **Social platform verification** — candidates on LinkedIn, X, Reddit,
  Instagram and others are classified and reported separately.
- **Trust scoring** over independent corroborating domains.
- **Tamper-evident anchoring** — an 11-leaf Merkle fingerprint of the evidence
  bundle, committed to a smart contract.
- **Local-only mode** — webcam and upload analysis with zero network calls,
  for when the image must not leave the machine.

---

## Architecture

One Python process serves both the API and the frontend. The face engine is
expensive to load (~7s) and is therefore a process-lifetime singleton — a
separate frontend dev server would mean two commands and one more thing to
fail during a demo.

```
                    ┌──────────────────────────────────┐
   browser  ◄──────►│  FastAPI + WebSocket (one proc)  │
   web/             │  src/tracelock/api               │
                    └────────────────┬─────────────────┘
                                     │
                         ┌───────────▼───────────┐
                         │  service/runner       │  orchestration,
                         │  stage progress       │  budget, cancellation
                         └───────────┬───────────┘
                                     │
   ┌──────────────┬──────────────────┼──────────────────┬───────────────┐
   │              │                  │                  │               │
┌──▼───────┐ ┌────▼──────┐ ┌─────────▼────────┐ ┌───────▼──────┐ ┌──────▼─────┐
│ ingest   │ │ discovery │ │ acquisition      │ │ face         │ │ chain      │
│ classify │ │ SerpAPI   │ │ SSRF guard, CAS  │ │ SCRFD +      │ │ Merkle +   │
│ resolve  │ │ Lens +    │ │ content-addressed│ │ ArcFace      │ │ Solidity   │
│ rank     │ │ Yandex    │ │ storage          │ │ 512-d        │ │ notary     │
└──────────┘ └───────────┘ └──────────────────┘ └──────┬───────┘ └────────────┘
                                                        │
                                          ┌─────────────▼─────────────┐
                                          │ verification + calibration│
                                          │ policy, bands, reasons    │
                                          └─────────────┬─────────────┘
                                                        │
                                                 ┌──────▼──────┐
                                                 │ evidence    │
                                                 │ aggregate,  │
                                                 │ trust score │
                                                 └─────────────┘
```

Detailed notes: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## How the pipeline works

### Candidate discovery → download → face analysis → verification → evidence

1. **Ingest.** The input is classified and resolved to a probe image. A local
   upload or webcam frame is analysed locally first; to enter live web search
   it must be published to temporary hosting, which requires explicit consent.

2. **Discovery.** Google Lens and Yandex Images are queried concurrently.
   Results are deduplicated, ranked, and marked *unverified by construction*.
   The search engine's opinion is a ranking signal, never evidence.

3. **Acquisition.** Each candidate image is downloaded through an SSRF guard
   that blocks localhost, private ranges, link-local addresses and cloud
   metadata endpoints, validating both the page URL and the resolved image
   URL. Bytes are stored content-addressed under `data/cas/` keyed by SHA-256,
   so the exact bytes that were measured survive link rot.

4. **Face analysis.** SCRFD detects faces in the downloaded bytes; ArcFace
   produces a 512-dimensional embedding. Quality gates reject images that are
   too small, corrupt, faceless, or ambiguous. **The embedding never leaves
   the process and is never written to the chain.**

5. **Verification.** Cosine similarity against the probe embedding is mapped
   through the calibration model to a probability, then banded.

6. **Evidence.** Verified candidates are aggregated by independent registrable
   domain, scored, and written as a JSON artifact. The artifact is fingerprinted
   and anchored.

Full detail: [docs/VERIFICATION_PIPELINE.md](docs/VERIFICATION_PIPELINE.md).

### Verified / Inconclusive / Rejected

The calibration model was fitted from real measured pairs and produces
`P(same identity | similarity) = sigmoid(16.5779 · s − 5.9037)`. Two
thresholds fall out of it:

| Band | Similarity | Status | Meaning |
|---|---|---|---|
| HIGH | `> 0.3528` | `VERIFIED_CANDIDATE` | Same person, on this evidence |
| INDETERMINATE | `0.2938 – 0.3528` | `INCONCLUSIVE` | Measured, but not enough to claim identity |
| LOW | `< 0.2938` | `REJECTED` | Different person (`LOW_FACE_SIMILARITY`) |

Candidates can also be rejected before ever reaching comparison —
`HTTP_ERROR`, `NO_FACE_DETECTED`, `NOT_AN_IMAGE`, `IMAGE_TOO_SMALL` and others.
Those are reported as counts, never as scored candidates, because no
comparison happened and showing one would imply a measurement that was never
made.

The inconclusive band is deliberate. Collapsing it into a yes/no would be
easier to demo and would be dishonest: a similarity of 0.32 genuinely does not
settle whether two photographs show the same person.

**Honest caveat on the calibration:** it was fitted on 5 genuine and 35
impostor pairs. AUC is 0.9886 and EER is 0.0, but the 95% confidence interval
on that EER is `[0.0000, 0.4345]`. The thresholds are indicative and are
labelled provisional in the code and the UI. This is recorded in
`data/calibration/model.json` under `confidence_note`.

### Trust scoring and anchoring

Trust is scored over **independent registrable domains** that carry verified
matches — ten reposts of one image on one site is one corroboration, not ten.
Duplicate content is detected by perceptual hash and reported rather than
silently merged.

The evidence bundle is canonicalised (RFC 8785 JCS), hashed into 11
domain-separated Merkle leaves, and the root is committed to an
`EvidenceNotary` Solidity contract. Only the 32-byte root and packed metadata
go on chain — no images, no embeddings, no personal data. A
`assert_no_biometric_leak` check enforces this before anything is submitted.

Anchoring runs in one of two modes, and the UI always states which:

- **LOCAL DEMO CHAIN** — an in-process EVM. Real cryptography, real contract,
  but *not publicly verifiable*. This is the default.
- **PUBLIC** — a real testnet, used only when `TL_RPC_URL` and
  `TL_PRIVATE_KEY` are both set and reachable.

The distinction is never blurred. A public network with missing credentials
resolves to local mode, because local is where the transaction will actually
execute.

---

## Installation

Requires **Python 3.11 or 3.12** (InsightFace does not yet build cleanly on
3.13).

```bash
git clone https://github.com/<your-username>/tracelock.git
cd tracelock
python -m venv .venv
```

```bash
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

```bash
pip install -e ".[dev]"
```

The InsightFace `buffalo_l` model pack (~330 MB) downloads on first use.

### Configuration

```bash
cp .env.example .env
```

Set `TL_SERPAPI_API_KEY` to run live web discovery — get a free key at
[serpapi.com](https://serpapi.com). Everything else has a working default;
local-only analysis and the local demo chain need no configuration at all.

---

## Running locally

```bash
python run.py
```

Then open <http://127.0.0.1:8000>. The API docs are at `/docs`.

The face model warms in the background (~7s) and the UI is usable
immediately. Without blockchain credentials TRACELOCK uses its local demo
chain and says so on screen.

Run the tests:

```bash
pytest
```

---

## Demo workflow

1. Open the UI and choose an input — upload, webcam, or paste a public image
   URL.
2. For a local upload or webcam frame, TRACELOCK analyses it locally and asks
   for explicit consent before publishing it to temporary hosting. Live
   reverse-image search is impossible without a publicly reachable URL, so
   this step is opt-in and stated plainly.
3. Choose **FAST** (stops once three independent publishers confirm) or
   **THOROUGH** (examines every candidate).
4. Watch the live stage progress over the WebSocket.
5. Read the result: trust score and verified matches, or a completed
   no-match with the full funnel.
6. Scroll to **Candidate verification evidence** to see the real candidates
   the search returned and what TRACELOCK decided about each.
7. Use the tamper test to mutate a field of the evidence bundle and watch
   fingerprint verification fail on the specific leaf.

Use **THOROUGH** when demonstrating, so verified and rejected candidates
appear together — FAST stops early and often shows verified only.

Step-by-step script: [docs/DEMO_GUIDE.md](docs/DEMO_GUIDE.md).

---

## Project structure

```
tracelock/
├── src/tracelock/
│   ├── acquisition/     SSRF-guarded download, content-addressed store
│   ├── api/             FastAPI app, WebSocket, routes
│   ├── calibration/     Platt fitting, metrics, model contract
│   ├── chain/           Merkle fingerprint, Solidity notary, mode resolution
│   ├── core/            config, models, run identity, rejection reasons
│   ├── discovery/       SerpAPI providers, temporary image hosting
│   ├── evidence/        aggregation, trust scoring
│   ├── face/            detection, embedding, quality, similarity
│   ├── ingest/          input classification, URL resolution, ranking
│   ├── service/         orchestration, caching, budget, streaming
│   └── verification/    policy, bands, perceptual hash, verifier
├── tests/               31 test modules
├── web/                 single-page frontend (no build step)
├── docs/                architecture, pipeline, demo guide, phase records
├── scripts/             pipeline and development tools
├── contracts/           EvidenceNotary.sol and its compiled ABI
└── data/                calibration model and examples (runtime data ignored)
```

---

## Technology stack

| Layer | Choice | Why |
|---|---|---|
| Face detection | SCRFD-10G (`det_10g.onnx`) | Accurate on small and angled faces |
| Face recognition | ArcFace ResNet50 (`w600k_r50.onnx`), 512-d | Strong open embedding model |
| Inference | ONNX Runtime, CPU | GPU setup costs more time than it saves at this volume |
| Backend | FastAPI + Uvicorn | Async, WebSocket, automatic API docs |
| Frontend | Plain HTML/CSS/JS | No build step; nothing to break on demo day |
| Search | SerpAPI (Google Lens, Yandex) | Official API access, no scraping |
| Blockchain | Solidity 0.8.24, web3.py, eth-tester | In-process EVM for demo, testnet-capable |
| Canonicalisation | RFC 8785 JCS | Byte-stable JSON so hashes are reproducible |

---

## Limitations

Stated plainly, because a tool that hides these is not trustworthy.

- **The calibration sample is small.** 5 genuine and 35 impostor pairs. The
  thresholds are provisional and labelled as such everywhere they appear.
- **Discovery only finds what the search engines index.** A photograph that
  was never published, or was published somewhere Google and Yandex do not
  crawl, returns nothing. That is a true negative, not a bug — but it is also
  the common case for private individuals.
- **Live search requires a publicly reachable URL.** Local uploads must be
  published to temporary hosting first, with consent. Local-only mode skips
  discovery entirely.
- **Platform coverage is uneven.** Sites that block automated fetching
  (some LinkedIn CDN paths, certain Twitter/X media hosts) may be discovered
  but fail to download. These appear as *not retrievable*, never as matches.
- **No live public testnet transaction has been performed.** The public chain
  path is implemented and tested against a mocked adapter; the local demo
  chain is what actually runs. The UI never claims otherwise.
- **Face verification is not identification.** A high similarity says two
  photographs are consistent with the same person. It is not a legal identity
  determination, and TRACELOCK's output should not be treated as one.
- **Run scoping is in-memory.** Candidate thumbnails are served per run from
  the process registry; restart the server and previous runs' images 404.

---

## Future improvements

- Refit calibration on a substantially larger labelled set to tighten the
  confidence interval and justify firmer thresholds.
- A live public testnet anchor, so verification does not depend on trusting
  the demo machine.
- Persist run metadata so candidate galleries survive a restart.
- Additional discovery providers (Bing Visual Search, TinEye) to reduce
  dependence on a single index.
- Optional GPU inference for larger candidate batches.
- Age and pose robustness testing — current behaviour on heavily aged or
  extreme-profile images is uncharacterised.

---

## Development history

The project was built in phases, and the working notes are kept because they
record why decisions were made:

- [docs/PHASE0_GATE_CHECKLIST.md](docs/PHASE0_GATE_CHECKLIST.md) — the
  day-zero viability gate. Before building anything, we checked whether
  candidate discovery from an image was possible at all in this environment.
- [docs/PHASE1_FACE_ENGINE.md](docs/PHASE1_FACE_ENGINE.md) — face engine
  validation, including the InsightFace-on-Python-3.12 risk.
- [docs/PHASE2_VERIFICATION.md](docs/PHASE2_VERIFICATION.md) — acquisition and
  the verification firewall, validated against real discovery output.
