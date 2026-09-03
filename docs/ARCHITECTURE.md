# Architecture

How TRACELOCK is put together and why it is shaped this way. For the
verification decision itself see
[VERIFICATION_PIPELINE.md](VERIFICATION_PIPELINE.md).

---

## One process

`python run.py` starts a single Uvicorn process that serves both the JSON API
and the static frontend from `web/`.

This is deliberate. The InsightFace model pack takes roughly 7 seconds to load
and about 326 MB of memory. It has to be a process-lifetime singleton, which
rules out anything that reloads per request. Serving the frontend from the
same process also means one command on demo day instead of two, and no CORS
configuration to get wrong.

The frontend has no build step — plain HTML, CSS and JavaScript. There is no
npm install to fail, no bundler cache to go stale, and the served bytes are
the bytes in the repository.

---

## Package map

| Package | Responsibility |
|---|---|
| `core` | Typed settings, shared models, run identity, the rejection-reason vocabulary |
| `ingest` | Classify an input, resolve it to an image, rank candidate images within a page |
| `discovery` | SerpAPI providers (Google Lens, Yandex), temporary image hosting |
| `acquisition` | SSRF-guarded fetching, content-addressed storage, provenance capture |
| `face` | Detection, embedding, quality gates, similarity |
| `calibration` | Platt fitting, metrics, the model contract |
| `verification` | Policy and bands, perceptual hashing, the per-candidate verifier |
| `evidence` | Aggregation across candidates, trust scoring |
| `chain` | Merkle fingerprint, Solidity notary, mode resolution, preflight |
| `service` | Orchestration, progress, caching, budget, cancellation, streaming |
| `api` | FastAPI routes, WebSocket, static serving |

Dependencies point inward. `core` imports nothing else from the project;
`api` and `service` sit at the outside.

---

## Request lifecycle

```
POST /api/investigate
        │
        ├─► service/runner: create Investigation, register in STORE
        │
        ├─► ingest:      classify + resolve the probe
        ├─► face:        embed the probe (singleton engine)
        │
        ├─► discovery:   Lens + Yandex, concurrently
        │
        └─► for each wave of candidates:
                acquisition ─► face ─► verification
                        │
                        └─► progress pushed over WebSocket

        ├─► evidence:    aggregate + trust score
        └─► chain:       fingerprint + anchor
```

Progress is streamed over `WS /api/ws/investigation/{run_id}`. If that socket
drops, the frontend falls back to polling `GET /api/investigation/{run_id}` —
a dropped socket must never leave the UI frozen, which it did until that
fallback was added.

---

## Key design decisions

### The face engine is a singleton

Loading `buffalo_l` per request would dominate every timing. It is loaded once
and shared. ONNX Runtime is pinned to `intra_op_num_threads = 4` with 8
worker threads — measured as the best configuration on the target hardware.
Letting ONNX use all cores while also running 8 workers oversubscribes the CPU
and is slower.

Only three of the five sub-models in the pack are loaded (`detection`,
`recognition`, `landmark_3d_68`); the others are never read.

### Content-addressed storage

Downloaded bytes are stored at `data/cas/blobs/<aa>/<sha256>` with a JSON
sidecar in `refs/` recording every URL that produced them. Three properties
follow:

- **Deduplication is explicit.** Two URLs with identical bytes store one blob
  and both source references. The duplicate is reported, not silently merged.
- **Evidence survives link rot.** The post gets deleted; the bytes remain, and
  the anchored hash proves they are unchanged.
- **Runs are reproducible.** A second run hits cache and produces identical
  hashes.

There is no database. A filesystem plus one sidecar per blob costs nothing
operationally at this scale.

### SSRF protection

Candidate URLs come from a third-party API and are treated as hostile. Both
the page URL and the resolved image URL are validated. Blocked: loopback,
private ranges, link-local, cloud metadata endpoints (169.254.169.254),
non-HTTP schemes, and `file://`.

Two layers exist because they answer different questions. `is_obviously_private`
is an offline pre-filter used early to reject clearly bad input without a DNS
round trip. `is_blocked_target` resolves DNS and is authoritative at fetch
time.

### Downloads have a watchdog

`iter_bytes(chunk_size)` blocks until a full chunk buffers, so a server
dribbling bytes slowly can run past a nominal timeout. A `threading.Timer`
closes the response at the deadline and partial bytes are discarded.

### Chain mode is derived, never configured

`resolve_chain_mode()` reports what the process *can actually do*. Public mode
requires `TL_RPC_URL` and `TL_PRIVATE_KEY` to both be present and the endpoint
to be reachable. Missing or unreachable resolves to `LOCAL_DEMO`, because
local is where the transaction will really execute and the interface has to
describe reality.

Connect and read timeouts are split (8s / 60s) so an unreachable RPC fails
fast and is classified as `unreachable` rather than a generic failure.

### Nothing biometric reaches the chain

`assert_no_biometric_leak` walks the payload before submission and raises if
it finds an embedding-shaped array. Only the 32-byte Merkle root and packed
metadata are written.

### Candidate media is served per run

`GET /api/investigation/{run_id}/candidate/{sha256}` serves a thumbnail from
the CAS, with two independent checks: the digest must be 64 hex characters
(so nothing can be walked out of the store), and it must appear in that run's
own results. The CAS is shared across every run the process has performed, so
path safety alone would still allow one run to read another's evidence.

The scope check runs before the filesystem is touched, so the 404-vs-200
difference cannot be used to probe what other runs hold.

### `cas_path` on the wire

`verification.results[].cas_path` is stripped from API responses — it is
internal layout the UI has no use for.

`evidence.items[].cas_path` is deliberately **kept**. The fingerprint hashes
`evidence["items"]` wholesale as the `tl:evidence_items` leaf, so removing the
field would make the served bundle stop reproducing the anchored Merkle root.
The path is relative (`data/cas/blobs/...`), never absolute. Tamper-evidence
is worth more than hiding a relative path.

---

## Frontend

`web/app.js` is a single script with no framework. It owns:

- the input flow (upload, webcam, URL, Drive, example)
- consent before any local image is published for live search
- WebSocket progress with polling fallback
- three result screens: verified, `completed_no_match`, and failure
- the candidate evidence gallery

State is minimal: `lastResult` holds the rendered result and `resultRunId`
holds the run that produced it. They are separate from `activeRunId`, which
gates cancellation and is cleared as soon as a run stops being cancellable —
before the renderers execute.

Assets are served with `no-store` and mtime-stamped URLs
(`app.js?v=<mtime>`), after a stale cached bundle silently ran old code across
several updates during development.
