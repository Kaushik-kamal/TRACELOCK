# Demo guide

A script for showing TRACELOCK live, and the honest answers to the questions
that get asked.

---

## Before you start

```bash
python run.py
```

- Wait for the face engine to warm (~7s). The badge reads **face engine
  ready**.
- Confirm **calibrated** appears in the status bar. Without it, verification
  refuses to score and the demo has nothing to show.
- Check `TL_SERPAPI_API_KEY` is set in `.env`, or live discovery will not run.
- Restart the server if it has been running across code changes. Static assets
  reload from disk, Python routes do not.

Have two images ready:

| | Purpose | Expected outcome |
|---|---|---|
| **A** | A well-indexed public figure | `complete` — verified matches, trust score, anchor |
| **B** | An ordinary personal photo | `completed_no_match` — lookalikes found and rejected |

**B is the more important demo.** Anyone can show a search engine finding a
famous face. Showing the tool refuse 22 strangers is the part that
demonstrates it is reasoning.

---

## Choose THOROUGH

FAST stops as soon as three independent publishers confirm a match, which
often means the rejected group is empty and the candidate gallery shows only
verified cards. THOROUGH examines every candidate, so verified, inconclusive
and rejected appear together.

Use THOROUGH whenever anyone is watching.

---

## The run

1. **Pick an input.** Upload, webcam, or paste a public image URL.

2. **Consent, if the image is local.** An uploaded or captured image has no
   public URL, and reverse-image search cannot reach it. TRACELOCK analyses it
   locally first, then asks before publishing it to temporary hosting.

   Say this out loud — it is a feature, not an apology. The tool refuses to
   silently upload a face.

3. **Start the investigation** and let the stages stream. The stage list is
   real progress from the pipeline, not an animation.

4. **Read the result.**

---

## Control A — a verified identity

Expect: verified matches across independent domains, a trust score with a
band, verified social posts, and an anchor.

Points worth making:

- **"The search engine's opinion was never trusted."** Every candidate was
  re-downloaded and re-measured locally. The panel *Was this a genuine live
  search?* names the engines that answered and how many candidates each
  returned.
- **Independent domains, not hit count.** Ten reposts on one site count once.
- **The anchor says LOCAL DEMO CHAIN.** Say so plainly — see below.

---

## Control B — a completed no-match

Expect `completed_no_match`, and this is the demo.

The screen leads with **LIVE SEARCH COMPLETED**, then the funnel:

```
25 discovered · 17 face-analysed · 0 verified · 1 inconclusive
16 rejected · 8 not retrievable
```

Then the highest similarity observed against the threshold required, and the
candidate gallery.

What to say:

> The search engines returned 25 visually similar faces. TRACELOCK downloaded
> them, ran its own face detection and embedding on the actual pixels, and
> measured each one against the probe. The closest was 0.3264 against a
> required 0.3528. Not one of them is the same person, so the tool reports
> zero matches — and shows you exactly who it rejected and by how much.

Then scroll to **Candidate verification evidence** and point at:

- **Real thumbnails.** These are the bytes that produced the score, served
  from the content-addressed store. Not re-fetched, not stock images.
- **Three distinct verdicts** — verified, inconclusive, rejected — each in its
  own colour.
- **The inconclusive card.** Usually the strongest moment. A candidate at
  0.3264 sits in the indeterminate band, and the tool says *"insufficient
  evidence to verify same identity"* rather than guessing.
- **No trust score, no anchor**, with the on-screen explanation that both
  require verified evidence.

---

## The tamper test

Open **Can this result be tampered with?** and run it. It mutates one field
of the evidence bundle, recomputes the fingerprint, and reports which of the
11 Merkle leaves no longer matches.

This is the chain-of-custody claim made concrete: not "trust us, it is
hashed", but "change anything and we can name what changed".

---

## Questions you will be asked

**"Is this really searching the web, or is it a fixture?"**
Real SerpAPI calls to Google Lens and Yandex Images, with `no_cache` set. The
*Was this a genuine live search?* panel lists which engines answered and how
many candidates each returned. Every thumbnail's bytes hash to the digest in
its own URL, so a substituted image could not keep the same address.

**"Is that a real blockchain?"**
By default, no — it is a local in-process EVM running the real
`EvidenceNotary` contract. Real Solidity, real Merkle root, real transaction,
but **not publicly verifiable**. The UI labels it **LOCAL DEMO CHAIN ·
ephemeral · not publicly verifiable** and never claims otherwise. Public
testnet anchoring is implemented and takes `TL_RPC_URL` and `TL_PRIVATE_KEY`,
but no live testnet transaction has been performed, so do not claim one.

**"How accurate is the threshold?"**
Fitted from measured pairs: AUC 0.9886, EER 0.0 — on 5 genuine and 35
impostor pairs. The 95% interval on that EER is `[0.0000, 0.4345]`, so the
thresholds are provisional and labelled that way in the code, the artifact and
the UI. Say this before someone else finds it.

**"Are you storing people's faces?"**
Candidate images are cached locally in the content-addressed store so the
evidence survives link rot. Nothing biometric ever reaches the chain — only a
32-byte Merkle root and packed metadata, enforced by a check that raises if an
embedding-shaped array appears in the payload. Runtime data is gitignored and
never committed.

**"What if it finds nothing?"**
That is Control B, and it is a supported terminal state rather than an error.

---

## If something goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| Thumbnails missing | Server predates a code change | Restart `run.py` |
| No candidates at all | Missing or exhausted SerpAPI key | Check `.env` |
| "calibrated" badge absent | `data/calibration/model.json` not found | Verify the file exists |
| Progress frozen | WebSocket dropped | The UI falls back to polling; give it a few seconds |
| Anchor step skipped | No verified evidence | Expected on `completed_no_match` |

If live discovery is unavailable entirely, the local-only path still
demonstrates face detection, quality gating and embedding without any network
call.
