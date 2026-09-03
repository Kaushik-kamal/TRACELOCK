# Phase 0 — Gate Validation Checklist

Work top to bottom. Do not start Phase 1 until section 4 is resolved.

---

## 1. Environment

- [ ] `python --version` reports 3.11.x or 3.12.x
- [ ] `.venv` created and activated (prompt shows `(.venv)`)
- [ ] `pip install -e ".[dev]"` completed without errors
- [ ] `pytest` passes — 92 tests, under 10 seconds, no network

## 2. Configuration

- [ ] `.env` exists (copied from `.env.example`)
- [ ] `TL_SERPAPI_API_KEY` is set to a real key
- [ ] `.env` is **not** tracked by git — confirm with `git status`
- [ ] SerpAPI dashboard shows remaining quota

## 3. Failure paths behave correctly

Each must produce the stated exit code. Check with `echo $LASTEXITCODE`
in PowerShell immediately after the command.

- [ ] Missing `--image-url` and no `--allow-upload` → **exit 2**, `SETUP ERROR`
- [ ] Invalid API key → **exit 2**, `SETUP ERROR` (*not* an architecture risk)
- [ ] Nonexistent `--image` path → **exit 3**, `INPUT ERROR`
- [ ] A `.jpg` containing text, not image data → **exit 3**, `INPUT ERROR`

If an invalid key reports `ARCHITECTURE RISK`, the error taxonomy is broken and
the gate's verdict cannot be trusted.

## 4. The real gate

Run against a **real face photo of a consenting subject** that is already
published somewhere public.

```powershell
python scripts\search_viability_gate.py --image data\probes\me.jpg --image-url "https://<public copy of the same image>"
```

### PASS requires all of

- [ ] Exit code **0**
- [ ] At least 1 candidate discovered
- [ ] At least 1 candidate with a usable image or thumbnail URL
- [ ] Run artifact written to `data/runs/search_gate_*.json`
- [ ] The artifact contains a non-empty `raw_response`
- [ ] Printed candidate URLs are plausible and **were not known in advance**

### FAIL if any of

- [ ] Provider returns 403 or 429 (blocked / rate limited)
- [ ] Zero candidates, or zero with usable media
- [ ] The response schema could not be parsed at all
- [ ] Automation is blocked by a CAPTCHA or requires manual intervention

## 5. If the gate fails

Work through these in order before concluding the architecture is unviable:

1. **Try `--engine yandex_images`.** Yandex is substantially better at face
   matching than Google Lens, which mostly finds the same image republished.
2. **Try a more widely published probe.** An image that exists in exactly one
   place has little for a reverse-image engine to find.
3. **Check quota** at <https://serpapi.com/dashboard>. Exhausted credits are a
   billing limit, not a hard block.
4. **Schema error?** The gate prints the top-level keys it actually received.
   Add the correct one to `RESULT_KEYS` in `src/tracelock/discovery/serpapi_lens.py`.
   Usually a two-minute fix.

Only if **several probes across both engines** return nothing does the discovery
strategy genuinely need rethinking — and that is exactly what this gate exists
to surface, on day zero rather than day five.

## 6. Anti-cheat audit

The gate is worthless if it can pass dishonestly. Verify by reading the code:

- [ ] No URL literal anywhere in `scripts/search_viability_gate.py` except
      documentation links in help text
- [ ] `no_cache=true` is sent on every query
- [ ] No import of anything from `tests/`
- [ ] Zero candidates produces FAIL, not PASS
- [ ] `raw_response` in the artifact is the provider's payload, unmodified
- [ ] `api_key` does not appear anywhere in a run artifact

## 7. Commit

- [ ] `git add -A && git commit -m "Phase 0: repository foundation and viability gate"`
- [ ] Confirm `data/runs/` and `.env` are absent from the commit

---

## Parallel day-zero task — do this tonight

Independent of the search gate, and the single highest-value five minutes in the
whole project:

- [ ] Create a testnet wallet
- [ ] Fund it on **Polygon Amoy**
- [ ] Fund it on **Base Sepolia** (fallback — the CDP faucet does not require a
      mainnet balance)
- [ ] Screenshot both balances

Faucets are rate-limited and periodically hostile. An empty wallet at 11pm on
demo night, with no time left, is the most common failure in this category of
project — and it has nothing to do with your code.

---

## Phase 1 readiness note

`insightface` ships as a source distribution and may need a compiler on Python
3.12. Before Phase 1 begins, verify:

```powershell
pip install insightface onnxruntime
python -c "import insightface; print(insightface.__version__)"
```

If that fails, install Python 3.11 and rebuild the venv against it — that is a
known-good path and costs 15 minutes. Discovering it during Phase 1 costs an
afternoon.
