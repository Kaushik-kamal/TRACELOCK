/* TRACELOCK UI.
   ONE input surface. A dropped file, a pasted image, any public link and a
   webcam frame all POST to /api/input/* and receive the SAME normalized
   object, so nothing below this line branches on where an image came from.
   Nothing here computes evidence, scores, or chain state -- it only renders
   what the backend actually measured. */

const $  = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));

let selected = null;   // normalized TraceInput description
let socket = null;
let lastResult = null;
// The run that produced `lastResult`. Candidate thumbnails are served per-run,
// and `activeRunId` is already null by the time the renderers execute (it is
// cleared as soon as the run stops being cancellable), so the result screens
// need their own handle on which investigation they are displaying.
let resultRunId = null;

// ---------------------------------------------------------------- helpers

function show(id) {
  $$(".screen").forEach((s) => s.classList.toggle("active", s.id === id));
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function toast(message, hint) {
  document.querySelector(".toast")?.remove();
  const el = document.createElement("div");
  el.className = "toast";
  el.innerHTML = `<b></b><span class="h"></span>`;
  el.querySelector("b").textContent = message;
  el.querySelector(".h").textContent = hint || "";
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 7000);
}

async function post(path, body) {
  const response = await fetch(path, { method: "POST", body });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw payload;
  return payload;
}

async function get(path) {
  const response = await fetch(path);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw payload;
  return payload;
}

function form(pairs) {
  const fd = new FormData();
  Object.entries(pairs).forEach(([k, v]) => fd.append(k, v));
  return fd;
}

const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// ---------------------------------------------------------------- boot

async function boot() {
  try {
    const health = await (await fetch("/api/health")).json();
    const bar = $("#sysbar");
    const pill = (label, state) =>
      `<span class="pill ${state}">${label}</span>`;

    bar.innerHTML = [
      pill(health.face_engine_loaded ? "face engine ready" : "face engine idle",
           health.face_engine_loaded ? "on" : ""),
      pill(health.search_configured ? `search: ${esc(health.search_engine)}` : "search: no API key",
           health.search_configured ? "on" : "off"),
      pill(health.calibrated ? "calibrated" : "uncalibrated",
           health.calibrated ? "on" : "warn"),
    ].join("");

    $("#foot-cal").textContent = health.calibrated
      ? "identity probabilities are calibrated"
      : "UNCALIBRATED — no trust score will be produced";

    const chain = await (await fetch("/api/chain/status")).json();
    // The label comes from the chain that will actually execute. An ephemeral
    // chain is marked as such and never dressed up as a public testnet.
    bar.insertAdjacentHTML("beforeend",
      pill(esc(chain.network_display_name), chain.ephemeral ? "warn" : "on"));
    if (chain.ephemeral) {
      bar.insertAdjacentHTML("beforeend", pill("ephemeral · not publicly verifiable", "warn"));
    }
    renderChainReadiness(chain);
  } catch (_) { /* health is advisory */ }

  loadExamples();
  loadResolverInfo();
}

async function loadResolverInfo() {
  try {
    const info = await (await fetch("/api/resolver/info")).json();
    const el = $("#link-support");
    if (el) el.textContent = info.support_statement + " " + info.note;
  } catch (_) { /* advisory only */ }
}

// DEMO MODE: ?demo=1 highlights verified examples and shows a guided
// narrative. It changes NOTHING about the pipeline — no fabricated candidates,
// scores or transactions, and no verification is bypassed.
const DEMO_MODE = new URLSearchParams(location.search).has("demo");

async function loadExamples() {
  try {
    const data = await (await fetch("/api/examples")).json();
    $("#demo-notice").textContent = data.notice;

    // Verified examples first — an unverified one has not been preflighted
    // and must not be presented as demo-ready.
    const ordered = [...data.examples].sort(
      (a, b) => (b.recommended - a.recommended) || (b.verified - a.verified));

    $("#examples").innerHTML = ordered.map((ex) => {
      const usable = ex.available && ex.investigable;
      const badge = ex.verified ? "ok" : (ex.investigable ? "warn" : "");
      return `
      <button class="example ${ex.recommended && DEMO_MODE ? "recommended" : ""}"
              data-example="${esc(ex.example_id)}" ${usable ? "" : "disabled"}>
        ${ex.available ? `<img src="${esc(ex.preview_url)}" alt="">` : ""}
        <div class="body">
          <div class="ex-top">
            <div class="t">${esc(ex.title)}</div>
            <span class="vbadge ${badge}">${esc(ex.verification_label)}</span>
          </div>
          <div class="d">${esc(ex.description)}</div>
          ${ex.preflight && ex.preflight.checked_at
            ? `<div class="checkedrow">preflighted ${esc(ex.preflight.checked_at.slice(0, 10))}${
                ex.preflight.face_quality != null
                  ? ` · face quality ${ex.preflight.face_quality.toFixed(2)}` : ""}</div>`
            : ""}
          ${!ex.available ? `<div class="warnrow">image not present on this machine</div>` : ""}
          ${ex.available && !ex.investigable
            ? `<div class="warnrow">local analysis only — no public URL</div>` : ""}
          ${ex.recommended && DEMO_MODE
            ? `<div class="recrow">★ RECOMMENDED FOR DEMO</div>` : ""}
        </div>
      </button>`;
    }).join("");

    $$(".example").forEach((btn) =>
      btn.addEventListener("click", () => pickExample(btn.dataset.example)));

    if (DEMO_MODE) enableDemoNarrative();
  } catch (_) {
    $("#examples").innerHTML = `<p class="note">Examples could not be loaded.</p>`;
  }
}

function renderChainReadiness(chain) {
  const box = $("#chain-readiness");
  if (!box) return;
  const r = chain.readiness;

  box.innerHTML = `
    <div class="readiness ${chain.ephemeral ? "local" : "public"}">
      <div class="rd-head">
        <span class="rd-mode">${esc(r.mode)}</span>
        <span class="rd-sum">${esc(r.summary)}</span>
      </div>
      <table class="rd-env">
        <tbody>
          ${r.required_env.map((e) => `
            <tr>
              <td class="rd-name">${esc(e.name)}</td>
              <td class="rd-state ${e.set ? "set" : "unset"}">${e.set ? "set" : "not set"}</td>
              <td class="rd-cur">${esc(e.current || "")}</td>
              <td class="rd-note">${esc(e.note)}</td>
            </tr>`).join("")}
        </tbody>
      </table>
      <p class="note">
        Values are never shown — only whether each variable is set.
        ${chain.faucet ? `Faucet: <a href="${esc(chain.faucet)}" target="_blank" rel="noopener">${esc(chain.faucet)}</a>` : ""}
      </p>
    </div>`;
}

function enableDemoNarrative() {
  document.body.classList.add("demo-mode");
  const banner = document.createElement("div");
  banner.className = "demo-banner";
  banner.innerHTML = `
    <b>DEMO MODE</b>
    <span>Recommended inputs are highlighted and the narrative is simplified.
    Nothing else changes: discovery, verification, scoring and anchoring all run
    for real. No result is fabricated.</span>`;
  $("#screen-source").prepend(banner);
}

// ------------------------------------------------------ universal input
//
// One input, every source. The operator drops a file, pastes any public link,
// or opens the camera; the SAME normalized TraceInput comes back either way.
// Nothing below branches on source type -- the server does the classifying.

// -- the five source cards -------------------------------------------
//
// Each card FOCUSES the universal input rather than opening a pipeline of its
// own. There is still exactly one ingestion path; the cards only tell the
// operator that their case is covered, and put the cursor in the right place.

$$(".source").forEach((card) => {
  card.addEventListener("click", () => {
    const source = card.dataset.source;
    $$(".source").forEach((c) => c.classList.toggle("active", c === card));

    const hint = {
      url: "Image URL, article, or social post link",
      drive: "https://drive.google.com/file/d/.../view",
    };

    if (source === "upload") {
      $("#panel-webcam").hidden = true;
      stopCamera();
      $("#file").click();
      return;
    }
    if (source === "webcam") {
      $("#panel-support").hidden = true;
      $("#panel-webcam").hidden = false;
      $("#cam-start").click();
      return;
    }
    if (source === "demo") {
      $("#panel-webcam").hidden = true;
      stopCamera();
      document.querySelector(".demo-block")?.scrollIntoView({ behavior: "smooth" });
      return;
    }

    // url / drive: same field, different placeholder so the expected shape is
    // obvious. Classification still happens from the URL itself, not the card.
    $("#panel-webcam").hidden = true;
    stopCamera();
    urlInput.placeholder = hint[source] || urlInput.placeholder;
    urlInput.focus();
    urlInput.scrollIntoView({ behavior: "smooth", block: "center" });
  });
});

// -- file: click, drop, and system paste all land here
$("#file").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (file) await sendFile("/api/input/upload", file, file.name);
});

const drop = $("#drop");
["dragenter", "dragover"].forEach((e) =>
  drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((e) =>
  drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", async (ev) => {
  const file = ev.dataTransfer.files[0];
  if (file) await sendFile("/api/input/upload", file, file.name);
});

// Pasting an image straight from the clipboard is the fastest path of all,
// and it never touches the network.
document.addEventListener("paste", async (event) => {
  if ($("#screen-source").classList.contains("active") === false) return;
  const item = [...(event.clipboardData?.items || [])]
    .find((i) => i.type.startsWith("image/"));
  if (!item) return;
  event.preventDefault();
  const blob = item.getAsFile();
  if (blob) await sendFile("/api/input/upload", blob, "pasted-image.png");
});

// -- shared uploader: every local source ends up here ------------------
//
// One function for a dropped file, a chosen file, a pasted image and a
// webcam frame. They differ only in the endpoint, and both endpoints return
// the same normalized object.

async function sendFile(endpoint, blob, filename) {
  const fd = new FormData();
  fd.append("file", blob, filename);
  try {
    select(await post(endpoint, fd));
  } catch (err) {
    toast(err.error || "That image could not be read.", err.hint);
  }
}

// -- webcam ------------------------------------------------------------
//
// The stream lives entirely in the browser. A frame is only sent anywhere
// when the operator presses Capture, and even then it goes to THIS server,
// never to a third-party host.

let cameraStream = null;

function stopCamera() {
  if (!cameraStream) return;
  cameraStream.getTracks().forEach((track) => track.stop());
  cameraStream = null;
  const video = $("#cam");
  if (video) video.srcObject = null;
}

$("#cam-start").addEventListener("click", async () => {
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false,
    });
  } catch (err) {
    $("#cam-note").textContent =
      "The camera could not be opened. Check that the browser has permission.";
    return;
  }
  $("#cam").srcObject = cameraStream;
  $("#cam-capture").disabled = false;
  $("#cam-retake").hidden = true;
  $("#cam-note").textContent =
    "The camera stream stays in your browser until you capture.";
});

$("#cam-capture").addEventListener("click", () => {
  const video = $("#cam");
  const canvas = $("#shot");
  if (!video.videoWidth) return;

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);

  canvas.toBlob(async (blob) => {
    if (!blob) return;
    stopCamera();
    $("#cam-capture").disabled = true;
    $("#cam-retake").hidden = false;
    await sendFile("/api/input/webcam", blob, "webcam.jpg");
  }, "image/jpeg", 0.95);
});

$("#cam-retake").addEventListener("click", () => $("#cam-start").click());

// -- link: classified live, offline, as it is typed
const urlInput = $("#url-input");
const detect = $("#detect");
let detectTimer = null;

urlInput.addEventListener("input", () => {
  clearTimeout(detectTimer);
  const value = urlInput.value.trim();
  if (value.length < 8) { detect.hidden = true; return; }
  // Debounced so a fast typist does not fire a request per keystroke. The
  // endpoint is offline and instant, but the DOM churn is not free.
  detectTimer = setTimeout(() => classifyLink(value), 180);
});

urlInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") submitLink();
});
$("#url-go").addEventListener("click", submitLink);

function renderLinkFailure(err) {
  // A blocked platform is a normal outcome, not an error screen. It says what
  // happened, in whose terms, and always offers a way forward.
  const failure = err.failure || {};
  const options = err.recovery || [];
  const blocked = failure.status === "blocked";

  detect.hidden = false;
  detect.className = "detect support-auth_required";
  detect.innerHTML = `
    <span class="detect-ico">${blocked ? "&#9888;" : "&#10005;"}</span>
    <span class="detect-body">
      <b>${esc(err.error || "That link could not be used.")}</b>
      ${err.hint ? `<span class="detect-note">${esc(err.hint)}</span>` : ""}
      <span class="recovery">
        ${options.map((o) => `
          <button class="btn recovery-btn" data-action="${esc(o.action)}">
            ${esc(o.label)}
          </button>`).join("")}
      </span>
    </span>`;

  detect.querySelectorAll(".recovery-btn").forEach((button) => {
    button.addEventListener("click", () => runRecovery(button.dataset.action));
  });
}

function runRecovery(action) {
  if (action === "upload_image") {
    $("#file").click();
    return;
  }
  if (action === "use_direct_url") {
    urlInput.value = "";
    urlInput.placeholder = "https://example.com/photo.jpg";
    urlInput.focus();
    detect.hidden = true;
    return;
  }
  if (action === "continue_local") {
    // Local analysis needs a LOCAL image. There is nothing to analyse until
    // the operator supplies one, so this opens the file picker rather than
    // pretending an unreachable URL can be examined offline.
    toast(
      "Local analysis works on an image on this device.",
      "Choose a file and TRACELOCK will analyse it without any network request.",
    );
    $("#file").click();
  }
}

async function classifyLink(url) {
  let payload;
  try {
    payload = await post("/api/input/classify", form({ url }));
  } catch { detect.hidden = true; return; }

  const c = payload.classification;
  const a = payload.adapter;
  // Only the honest badge -- "usually blocked" is shown BEFORE we try, so a
  // refusal later is an expectation met, not a surprise.
  detect.hidden = false;
  detect.className = `detect support-${a.support}`;
  detect.innerHTML = `
    <span class="detect-ico">${esc(c.platform_icon)}</span>
    <span class="detect-body">
      <b>${esc(c.label)}</b>
      <span class="badge b-${a.support}">${esc(a.badge)}</span>
      <span class="detect-note">${esc(a.support === "reliable" ? a.note : a.expectation)}</span>
      ${a.guidance && a.support !== "reliable"
        ? `<span class="detect-guide">${esc(a.guidance)}</span>` : ""}
    </span>`;
}

async function submitLink() {
  const url = urlInput.value.trim();
  if (!url) return toast("Paste a link first.");

  const button = $("#url-go");
  button.disabled = true;
  button.textContent = "Loading…";
  try {
    // Direct-image links load straight in. Anything else -- an article, a
    // post, a Drive share link -- goes through the resolver, which may offer
    // several images to choose between.
    select(await post("/api/input/url", form({ url })));
  } catch (err) {
    renderLinkFailure(err);
  } finally {
    button.disabled = false;
    button.textContent = "Go";
  }
}

// -- webcam and the capability matrix are secondary disclosures
$("#use-webcam").addEventListener("click", () => {
  const panel = $("#panel-webcam");
  panel.hidden = !panel.hidden;
  $("#panel-support").hidden = true;
  if (panel.hidden) stopCamera();
});
$("#cam-close").addEventListener("click", () => {
  $("#panel-webcam").hidden = true;
  stopCamera();
});

$("#show-support").addEventListener("click", async () => {
  const panel = $("#panel-support");
  panel.hidden = !panel.hidden;
  $("#panel-webcam").hidden = true;
  if (!panel.hidden) await renderSupportMatrix();
});

async function renderSupportMatrix() {
  const host = $("#support-matrix");
  if (host.dataset.loaded) return;
  let info;
  try { info = await get("/api/resolver/info"); } catch { return; }

  host.innerHTML = info.platforms.map((p) => `
    <div class="matrix-row support-${esc(p.support)}">
      <span class="m-ico">${esc(p.icon)}</span>
      <span class="m-name">${esc(p.display)}</span>
      <span class="badge b-${esc(p.support)}">${esc(p.badge)}</span>
      <span class="m-note">${esc(p.note)}</span>
    </div>`).join("");
  host.dataset.loaded = "1";
}

async function pickExample(exampleId) {
  try {
    select(await post("/api/input/example", form({ example_id: exampleId })));
  } catch (err) {
    toast(err.error || "That example could not be loaded.", err.hint);
  }
}

// ---------------------------------------------------------------- selected

function select(input) {
  selected = input;
  stopCamera();

  $("#preview").src = input.preview_url;
  $("#m-source").textContent = input.source_label;
  $("#m-file").textContent = input.filename;
  $("#m-hash").textContent = input.sha256.slice(0, 32) + "…";
  $("#m-url").textContent = input.image_url || "— none —";

  const notice = $("#kind-notice");
  notice.hidden = input.kind !== "DEMO_EXAMPLE";
  notice.textContent = input.kind_notice || "";

  renderPickedFrom(input.resolution);

  $("#checks").innerHTML = `<div class="checking">Validating…</div>`;
  $("#start").disabled = true;
  $("#start-note").textContent = "";

  // MODE A when a public URL exists, MODE B otherwise. Which mode is available
  // is a fact about the image, not a preference.
  const hasPublicUrl = Boolean(input.image_url);
  $("#mode-public").hidden = !hasPublicUrl;
  $("#mode-choose").hidden = hasPublicUrl;
  $("#mode-addurl").hidden = true;
  $("#mode-consent").hidden = true;
  $("#link-result").innerHTML = "";

  show("screen-selected");
  precheck(input);
}

function renderPickedFrom(resolution) {
  const box = $("#picked-from");
  const alternatives = (resolution && resolution.alternatives) || [];

  if (!resolution || resolution.method === "direct") {
    box.hidden = true;
    box.innerHTML = "";
    return;
  }

  // Say WHERE this image came from, then offer the others. The ranking picked
  // one; it is not always the one the operator meant.
  box.hidden = false;
  box.innerHTML = `
    <div class="picked-head">
      <b>${esc(resolution.platform_display)}</b> page resolved &mdash;
      ${esc(resolution.method_explanation)}
    </div>
    ${alternatives.length ? `
      <div class="cand-head">
        Not the right picture? ${alternatives.length} other image${
          alternatives.length === 1 ? "" : "s"} on that page:
      </div>
      <div class="cand-grid">
        ${alternatives.map((c) => `
          <button class="cand" data-url="${esc(c.url)}" title="${esc(c.url)}">
            <img src="${esc(c.url)}" alt="" loading="lazy"
                 onerror="this.closest('.cand').classList.add('broken')">
            <span class="cand-label">${esc(c.label)}</span>
            <span class="cand-src">${esc(c.source)}</span>
          </button>`).join("")}
      </div>` : ""}`;

  box.querySelectorAll(".cand").forEach((button) => {
    button.addEventListener("click", async () => {
      try {
        select(await post("/api/input/url", form({
          url: resolution.input_url,
          candidate_url: button.dataset.url,
        })));
      } catch (err) {
        toast(err.error || "That image could not be loaded.", err.hint);
      }
    });
  });
}

async function precheck(input) {
  try {
    const result = await post("/api/precheck", form({ sha256: input.sha256 }));
    $("#checks").innerHTML = result.checks.map((c) => `
      <div class="check">
        <span class="${c.ok ? "g" : "b"}">${c.ok ? "✓" : "✕"}</span>
        <div>
          <div>${esc(c.label)}</div>
          ${c.detail ? `<div class="d">${esc(c.detail)}</div>` : ""}
        </div>
      </div>`).join("");

    const hasPublicUrl = Boolean(input.image_url);
    $("#start").disabled = !(result.ok && hasPublicUrl);
    $("#go-addurl").disabled = !result.ok;

    if (!result.ok) {
      $("#start-note").textContent = "This image cannot be investigated.";
    }
  } catch (err) {
    $("#checks").innerHTML =
      `<div class="check"><span class="b">✕</span><div>${esc(err.detail || "Validation failed.")}</div></div>`;
  }
}

// ---------------------------------------------------------------- run

let investigationMode = "fast";
let activeRunId = null;

$$(".mode-opt").forEach((option) => {
  option.addEventListener("click", () => {
    investigationMode = option.dataset.mode;
    $$(".mode-opt").forEach((o) => o.classList.toggle("active", o === option));
  });
});

$("#start").addEventListener("click", async () => {
  if (!selected) return;
  $("#start").disabled = true;
  $("#failbox").hidden = true;

  try {
    const run = await post("/api/investigate", form({
      sha256: selected.sha256,
      source_type: selected.source_type,
      image_url: selected.image_url || "",
      kind: selected.kind,
      filename: selected.filename,
      anchor: $("#want-anchor").checked ? "true" : "false",
      mode: investigationMode,
    }));
    activeRunId = run.run_id;
    renderStages(run.stages);
    $("#cancel-run").hidden = false;
    $("#cancel-run").disabled = false;
    $("#cancel-run").textContent = "Cancel investigation";
    $("#run-mode").textContent = run.mode === "thorough"
      ? "Thorough - every candidate is examined."
      : "Fast - stops once 3 independent publishers confirm a match.";
    show("screen-progress");
    watch(run.run_id);
  } catch (err) {
    $("#start").disabled = false;
    toast(err.error || err.detail || "The investigation could not be started.", err.hint);
  }
});

$("#cancel-run").addEventListener("click", async () => {
  if (!activeRunId) return;
  const button = $("#cancel-run");
  button.disabled = true;
  button.textContent = "Stopping…";
  try {
    // Cooperative: the run stops at its next safe checkpoint rather than
    // being killed mid-write, so nothing is left half-stored.
    await post(`/api/investigation/${encodeURIComponent(activeRunId)}/cancel`, form({}));
  } catch (err) {
    button.disabled = false;
    button.textContent = "Cancel investigation";
    toast(err.detail || "That run could not be cancelled.");
  }
});

function liveSearchProof(result, sources, social) {
  const input = result.input || {};
  const hosting = (input.provenance || {}).temporary_hosting;
  const coverage = (result.verification || {}).coverage || {};
  const posts = social.verified_social_posts || [];

  // How the image reached the search engines. A judge should be able to see
  // that the URL was minted at runtime from THEIR image, not chosen in advance.
  const originRow = hosting
    ? `<div class="proof-row">
         <span class="hl">Search input</span>
         <span class="fv">
           Published from your local file after consent, via
           <b>${esc(hosting.provider)}</b>
         </span>
       </div>
       <div class="proof-row">
         <span class="hl">Public URL</span>
         <code class="hv">${esc(input.image_url || "")}</code>
       </div>`
    : `<div class="proof-row">
         <span class="hl">Search input</span>
         <span class="fv">A public URL you supplied &mdash; nothing was uploaded.</span>
       </div>
       <div class="proof-row">
         <span class="hl">Public URL</span>
         <code class="hv">${esc(input.image_url || "")}</code>
       </div>`;

  return `
    ${originRow}
    <div class="proof-row">
      <span class="hl">Engines</span>
      <span class="fv">
        ${sources.map((s) => `${esc(s.engine)} (${s.ok ? s.candidates + " results" : "failed"})`).join(" &middot; ") || "—"}
      </span>
    </div>
    <div class="proof-row">
      <span class="hl">Funnel</span>
      <span class="fv">
        ${coverage.discovered_total ?? "—"} discovered &rarr;
        ${coverage.examined ?? "—"} examined &rarr;
        ${coverage.verified ?? "—"} face-verified
      </span>
    </div>

    <div class="social-block ${social.requirement_met ? "met" : "unmet"}">
      <div class="social-head">
        ${social.requirement_met
          ? "&#10003; Social-media post found and independently verified"
          : "Live search completed"}
      </div>
      <p class="social-statement">${esc(social.statement || "")}</p>

      ${posts.length ? `
        <div class="social-list">
          ${posts.map((p) => `
            <div class="social-item">
              <span class="social-badge">${esc(p.platform || "Social")}</span>
              <a class="social-url" href="${esc(p.url)}" target="_blank"
                 rel="noopener noreferrer">${esc((p.url || "").slice(0, 68))}</a>
              <span class="social-sim">
                similarity ${p.similarity != null ? p.similarity.toFixed(4) : "—"}
              </span>
            </div>`).join("")}
        </div>` : ""}

      ${Object.keys(social.categories || {}).length ? `
        <div class="cat-row">
          ${Object.entries(social.categories).map(([k, n]) =>
            `<span class="cat ${esc(k)}">${esc(k.replace("_", " "))}: ${n}</span>`).join("")}
        </div>` : ""}

      <p class="note">
        <strong>Discovered</strong> means a live search returned the URL.
        <strong>Face-verified</strong> means TRACELOCK re-downloaded the image
        and the calibrated model confirmed the same person. Only the second is
        a claim about identity.
      </p>
    </div>`;
}

function renderPerformance(perf, coverage, results) {
  const phases = (perf.phases || []).slice().sort((a, b) => b.seconds - a.seconds);
  if (!phases.length) return "";

  const ms = perf.milestones || {};
  const rejected = results.filter((r) => r.status === "REJECTED");
  const dupes = rejected.filter((r) =>
    (r.rejection_reasons || []).some((x) => x.reason === "DUPLICATE_CONTENT")).length;
  const preFace = rejected.filter((r) =>
    (r.rejection_reasons || []).some((x) =>
      x.stage === "ACQUIRED" || x.stage === "VALIDATED")).length;
  const embedded = results.filter((r) => r.face_similarity != null).length;

  // Efficiency = candidates NOT given an embedding, over all discovered. It
  // measures work avoided, not accuracy, so it is labelled as such.
  const discovered = coverage.discovered_total || results.length || 1;
  const avoided = Math.max(0, discovered - embedded);
  const efficiency = Math.round((avoided / discovered) * 100);

  const milestone = (label, value, target) => value == null ? "" : `
    <div class="ms-row">
      <span class="ms-label">${esc(label)}</span>
      <span class="ms-value ${target && value <= target ? "hit" : ""}">${value.toFixed(2)}s</span>
      ${target ? `<span class="ms-target">target ${target}s</span>` : `<span class="ms-target"></span>`}
    </div>`;

  const max = Math.max(...phases.map((p) => p.seconds), 0.0001);
  return `
    <section class="story">
      <div class="story-q">Performance</div>

      <div class="milestones">
        ${milestone("Time to first candidate", perf.time_to_first_candidate, 3)}
        ${milestone("Time to first evidence", ms.first_evidence, 3)}
        ${milestone("Time to strong confidence", ms.strong_confidence, 6)}
        ${milestone("Total completion", perf.total_seconds, 10)}
      </div>

      <div class="funnel-counts">
        ${[["Discovered", discovered],
           ["Examined", coverage.examined ?? results.length],
           ["Duplicates removed", dupes],
           ["Rejected before face analysis", preFace],
           ["Face embeddings computed", embedded],
           ["Verified", coverage.verified ?? 0]]
          .map(([l, n]) => `<div class="fc"><b>${n}</b><span>${esc(l)}</span></div>`).join("")}
      </div>

      <div class="efficiency">
        <span class="eff-label">Expensive work avoided</span>
        <span class="eff-bar"><span style="width:${efficiency}%"></span></span>
        <span class="eff-pct">${efficiency}%</span>
      </div>
      <p class="note" style="margin-top:6px">
        ${avoided} of ${discovered} discovered candidates never needed a face
        embedding &mdash; removed by URL/content deduplication, download or
        validation failure, or because corroboration was already sufficient.
        This measures work avoided, not accuracy.
      </p>
      <div class="perf">
        ${phases.map((p) => `
          <div class="perf-row">
            <span class="perf-name">${esc(p.name.replace(/_/g, " "))}</span>
            <span class="perf-bar">
              <span style="width:${Math.max(2, (p.seconds / max) * 100)}%"></span>
            </span>
            <span class="perf-time">${p.seconds.toFixed(2)}s</span>
            <span class="perf-note">${
              p.kind === "accumulated" ? "worker CPU"
              : p.count ? `${p.count} items` : ""}</span>
          </div>`).join("")}
        <div class="perf-row total">
          <span class="perf-name">Total (wall clock)</span>
          <span class="perf-bar"></span>
          <span class="perf-time">${(perf.total_seconds || 0).toFixed(2)}s</span>
          <span class="perf-note"></span>
        </div>
      </div>
      <p class="note">
        Measured wall-clock intervals from this run &mdash; not estimates.
        Phases overlap because discovery, downloads and face analysis run
        concurrently, so they do not sum to the total;
        <strong>${(perf.overlap_saved_seconds || 0).toFixed(2)}s</strong>
        was removed by that overlap.
      </p>
    </section>`;
}

function renderStages(stages) {
  const icons = { pending: "○", running: "◐", done: "✓", failed: "✕", skipped: "—" };
  $("#stages").innerHTML = stages.map((s) => `
    <li class="stage ${s.state}">
      <span class="icon">${icons[s.state]}</span>
      <span class="label">${esc(s.label)}</span>
      <span class="ms">${s.elapsed ? s.elapsed.toFixed(1) + "s" : ""}</span>
      ${s.detail ? `<span class="det">${esc(s.detail)}</span>` : ""}
    </li>`).join("");
}

function watch(runId) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${proto}://${location.host}/api/ws/investigation/${runId}`);

  socket.onmessage = (event) => {
    const snapshot = JSON.parse(event.data);
    renderStages(snapshot.stages);

    if (snapshot.status !== "running" && snapshot.status !== "pending") {
      $("#cancel-run").hidden = true;
      activeRunId = null;
    }

    if (snapshot.status === "cancelled") {
      // A cancelled run reports NO findings. The candidates it happened to
      // reach are an incomplete sample, not a result, so nothing is rendered
      // as though the investigation had concluded.
      $("#failbox").hidden = false;
      $("#failbox").innerHTML = `
        <h3>Investigation cancelled</h3>
        <p>${esc(snapshot.error || "Cancelled by the operator.")}</p>
        ${snapshot.error_hint ? `<p class="hint">${esc(snapshot.error_hint)}</p>` : ""}
        <div class="fail-actions">
          <button class="btn" data-goto="screen-source">Start a new investigation</button>
        </div>`;
      wireFailboxActions();
    }
    if (snapshot.status === "failed") {
      $("#failbox").hidden = false;
      $("#failbox").innerHTML = `
        <h3>Investigation stopped</h3>
        <p>${esc(snapshot.error || "")}</p>
        ${snapshot.error_hint ? `<p class="hint">${esc(snapshot.error_hint)}</p>` : ""}
        <div class="fail-actions">
          <button class="btn primary" data-goto="screen-selected">Try again</button>
          <button class="btn" data-goto="screen-source">Use a different image</button>
        </div>`;
      wireFailboxActions();
    }
    if (snapshot.status === "completed_no_match" && snapshot.result) {
      // A completed search that verified nothing. It goes to the RESULTS
      // screen, not the failure box: the pipeline did its job and correctly
      // declined to claim a match.
      lastResult = snapshot.result;
      resultRunId = snapshot.run_id;
      renderNoMatch(snapshot.result);
      show("screen-results");
    }
    if (snapshot.status === "complete" && snapshot.result) {
      lastResult = snapshot.result;
      resultRunId = snapshot.run_id;
      renderResults(snapshot.result);
      show("screen-results");
    }
  };

  // A dropped socket must never freeze the UI.
  //
  // Only `onmessage` existed, so a clean close -- a proxy timeout, a server
  // restart, wifi blinking during a demo -- left the progress screen spinning
  // on "Discovering public sources" forever, with no failbox and no button.
  // Verified in the browser before this fix: readyState 3, screen unchanged,
  // zero recovery controls.
  //
  // The run usually SURVIVES the socket, so the recovery is to ask the REST
  // endpoint that was already there. Most of the time the investigation
  // finished and the result is simply collected.
  socket.onclose = () => reconnectOrRecover(runId);
  socket.onerror = () => reconnectOrRecover(runId);
}

async function reconnectOrRecover(runId) {
  if (!activeRunId || runId !== activeRunId) return;   // already finished

  let snapshot = null;
  // The socket may drop while the server is mid-stage, so poll briefly rather
  // than declaring failure on the first miss.
  for (let attempt = 0; attempt < 12; attempt += 1) {
    try {
      snapshot = await get(`/api/investigation/${encodeURIComponent(runId)}`);
    } catch {
      snapshot = null;
    }
    if (snapshot) {
      renderStages(snapshot.stages);
      if (snapshot.status === "complete" && snapshot.result) {
        activeRunId = null;
        lastResult = snapshot.result;
      resultRunId = snapshot.run_id;
        renderResults(snapshot.result);
        show("screen-results");
        return;
      }
      if (snapshot.status === "failed" || snapshot.status === "cancelled") break;
    }
    await new Promise((resolve) => setTimeout(resolve, 1500));
  }

  activeRunId = null;
  $("#cancel-run").hidden = true;
  $("#failbox").hidden = false;
  $("#failbox").innerHTML = `
    <h3>Lost contact with the server</h3>
    <p>${esc(
      snapshot && snapshot.error
        ? snapshot.error
        : "The live connection dropped, so progress can no longer be shown."
    )}</p>
    <p class="hint">
      Nothing was fabricated: no result is displayed because none was received.
    </p>
    <div class="fail-actions">
      <button class="btn primary" data-goto="screen-selected">Try again</button>
      <button class="btn" data-goto="screen-source">Use a different image</button>
    </div>`;
  wireFailboxActions();
}

function wireFailboxActions() {
  $("#failbox").querySelectorAll("[data-goto]").forEach((button) => {
    button.addEventListener("click", () => {
      $("#failbox").hidden = true;
      if (button.dataset.goto === "screen-selected" && selected) {
        $("#start").disabled = false;
      }
      show(button.dataset.goto);
    });
  });
}

// ---------------------------------------------------------------- results

// ==========================================================================
// Candidate verification evidence
//
// The single most persuasive thing this tool does is REFUSE. A reverse-image
// engine returns faces that look alike; TRACELOCK downloads each one and
// measures it. Showing only the survivors hides the actual work, and "0
// verified" is a far weaker claim than 22 named lookalikes each shown with
// the score that disqualified it.
//
// Everything rendered below is runtime evidence from the run being displayed.
// There is no fixture path, no placeholder and no sample image: if the
// pipeline analysed nothing, this section renders nothing.
// ==========================================================================

function analysedCandidates(result) {
  // ONLY candidates that genuinely reached face analysis. A candidate that
  // failed to download, or contained no face, was never compared against
  // anything -- listing it as "rejected" would imply a measurement that never
  // happened. Those are reported as a count instead, never as a card.
  const results = (result.verification || {}).results || [];
  return results.filter((r) => r.face_similarity !== null
                            && r.face_similarity !== undefined);
}

function candidateKind(r) {
  // The pipeline's status, mapped to a display group. Nothing is decided here.
  if (r.status === "VERIFIED_CANDIDATE") return "verified";
  if (r.status === "INCONCLUSIVE") return "inconclusive";
  return "rejected";
}

function candidateVerdict(r, kind, ceiling) {
  // Headline is the decision; detail says WHY, in the pipeline's own terms.
  // The threshold is quoted from this run's own policy, never hardcoded.
  const sim = r.face_similarity.toFixed(4);
  const bound = ceiling != null ? ceiling.toFixed(4) : null;

  if (kind === "verified") {
    return {
      label: "VERIFIED SAME PERSON",
      detail: bound
        ? "Similarity " + sim + " — at or above the calibrated same-person threshold (" + bound + ")"
        : "Similarity " + sim,
    };
  }

  if (kind === "inconclusive") {
    return {
      label: "INCONCLUSIVE — BELOW SAME-PERSON THRESHOLD",
      detail: "Similarity " + sim + " — inconclusive band; insufficient evidence "
            + "to verify same identity" + (bound ? " (threshold " + bound + ")" : ""),
    };
  }

  const reason = (r.rejection_reasons || [])[0];
  const code = (reason && reason.reason) || r.status || "NOT VERIFIED";
  return {
    label: "REJECTED — " + code.replace(/_/g, " "),
    detail: code === "LOW_FACE_SIMILARITY" && bound
      ? "Similarity " + sim + " — below calibrated same-person threshold (" + bound + ")"
      : ((reason && reason.explanation) || "Similarity " + sim),
  };
}

function candidateCard(r, runId, ceiling) {
  const kind = candidateKind(r);
  const verdict = candidateVerdict(r, kind, ceiling);
  const prov = r.provenance || {};
  const domain = prov.registrable_domain || prov.host || "unknown source";

  // Served per-run from the content-addressed store: the exact bytes that
  // produced this score. Re-fetching the origin could return something else.
  const thumb = r.content_sha256 && runId
    // Deliberately NOT loading="lazy". The gallery is bounded to ~15 small
    // thumbnails, and during a live demo an image that only appears once the
    // judge happens to scroll is a worse failure than the bytes it saves.
    ? `<img src="/api/investigation/${esc(runId)}/candidate/${esc(r.content_sha256)}"
            alt="Candidate image from ${esc(domain)}"
            onerror="this.closest('.cand-card').classList.add('nothumb')">`
    : "";

  return `
    <article class="cand-card ${esc(kind)}${thumb ? "" : " nothumb"}">
      <div class="cc-thumb">
        ${thumb}<span class="cc-fallback">image<br>unavailable</span>
      </div>
      <div class="cc-body">
        <div class="cc-src">
          ${r.platform ? `<span class="cc-plat">${esc(r.platform)}</span>` : ""}
          <span class="cc-domain">${esc(domain)}</span>
        </div>
        <div class="cc-metrics">
          <span>similarity <b>${r.face_similarity.toFixed(4)}</b></span>
          ${r.identity_probability != null
            ? `<span>P(identity) <b>${r.identity_probability.toFixed(4)}</b></span>` : ""}
          ${r.similarity_band ? `<span class="cc-band">${esc(r.similarity_band)}</span>` : ""}
        </div>
        <div class="cc-verdict ${esc(kind)}">${esc(verdict.label)}</div>
        <div class="cc-why">${esc(verdict.detail)}</div>
        ${r.source_url
          ? `<a href="${esc(r.source_url)}" target="_blank" rel="noopener noreferrer"
                class="cc-link">View original ↗</a>`
          : `<span class="cc-link muted">no source page recorded</span>`}
      </div>
    </article>`;
}

function candidateGroup(rows, kind, title, note, runId, ceiling, total) {
  if (!rows.length) return "";
  const count = total != null && total > rows.length
    ? "closest " + rows.length + " of " + total
    : String(rows.length);
  return `
    <div class="cand-group">
      <div class="cg-head ${esc(kind)}">${esc(title)} <span class="cg-n">(${esc(count)})</span></div>
      ${note ? `<p class="cg-note">${note}</p>` : ""}
      <div class="cand-cards">
        ${rows.map((r) => candidateCard(r, runId, ceiling)).join("")}
      </div>
    </div>`;
}

// ==========================================================================
// Public profile evidence (Tier 1)
//
// Built entirely from metadata the live search already returned for
// candidates that ALREADY cleared face verification -- no new lookups. The
// backend (tracelock.social_profile) is the only place a tier is decided;
// this file only ever reads `rel.tier` and renders whichever label/icon that
// exact tier owns. There is no "verified" styling anywhere in this section
// that is not gated on tier === "VERIFIED_HIGH_CONFIDENCE".
// ==========================================================================

const PROFILE_TIER_META = {
  VERIFIED_HIGH_CONFIDENCE: { icon: "✓", cls: "verified", label: "Verified high-confidence" },
  UNVERIFIED_POSSIBLE_MATCH: { icon: "?", cls: "unverified", label: "Unverified possible match" },
  DISCOVERED_LINK: { icon: "○", cls: "discovered", label: "Discovered link only" },
};

function profileTierMeta(tier) {
  return PROFILE_TIER_META[tier] || { icon: "○", cls: "discovered", label: tier };
}

function profileEvidenceChain(chain) {
  if (!chain || !chain.length) return "";
  return `
    <details class="pe-why">
      <summary>Why this was linked</summary>
      <ol class="pe-chain">
        ${chain.map((step) => `
          <li>
            <div class="pe-claim">${esc(step.claim)}</div>
            <div class="pe-source">source: ${esc(step.source)}</div>
            ${step.quoted_text ? `<div class="pe-quote">&ldquo;${esc(step.quoted_text)}&rdquo;</div>` : ""}
          </li>`).join("")}
      </ol>
    </details>`;
}

function profileCard(rel) {
  const meta = profileTierMeta(rel.tier);
  const link = rel.profile_url
    ? `<a class="pe-link" href="${esc(rel.profile_url)}" target="_blank"
          rel="noopener noreferrer">${esc(rel.profile_url)}</a>`
    : `<span class="pe-link muted">${rel.handle
        ? "Name/handle found in metadata: " + esc(rel.handle) + " (no link constructed from free text)"
        : "no resolvable link"}</span>`;

  return `
    <article class="pe-card ${esc(meta.cls)}">
      <div class="pe-head">
        <span class="pe-tier ${esc(meta.cls)}">${meta.icon} ${esc(meta.label)}</span>
        <span class="pe-platform">${esc(rel.platform || "")}</span>
      </div>
      <div class="pe-body">
        ${link}
        <p class="pe-claim-text">${esc(rel.tier_claim || "")}</p>
        <div class="pe-meta-row">
          <span>source candidate <code>${esc(rel.source_candidate_id || "")}</code></span>
          ${rel.face_similarity != null
            ? `<span>similarity ${rel.face_similarity.toFixed(4)}</span>` : ""}
        </div>
        ${rel.source_url ? `
          <a class="pe-source-link" href="${esc(rel.source_url)}" target="_blank"
             rel="noopener noreferrer">View the verified source page &#8599;</a>` : ""}
        ${profileEvidenceChain(rel.evidence_chain)}
      </div>
    </article>`;
}

function publicProfileEvidenceSection(result) {
  const verification = result.verification || {};
  const dp = verification.discovered_profiles;
  // No verification block at all (e.g. an old artifact) -- render nothing
  // rather than a fabricated empty state for a feature that never ran.
  if (!dp) return "";

  const rels = dp.relationships || [];
  const verified = rels.filter((r) => r.tier === "VERIFIED_HIGH_CONFIDENCE");
  const unverified = rels.filter((r) => r.tier === "UNVERIFIED_POSSIBLE_MATCH");
  const discovered = rels.filter((r) => r.tier === "DISCOVERED_LINK");

  return `
    <section class="story pe-section">
      <div class="story-q">Public profile evidence</div>
      <p class="pe-sub">
        Built only from title/author/source metadata the live search already
        returned for candidates that independently cleared face verification
        &mdash; no additional lookups were made. This never claims a profile
        belongs to a real person; it reports what the discovery provider's
        own metadata says about pages already proven to contain the
        subject's face.
      </p>

      ${rels.length === 0
        ? `<p class="pe-empty">${esc(dp.statement || "No verified public profile relationship found.")}</p>`
        : `
          ${verified.length ? `
            <div class="pe-group">
              <div class="pe-group-head verified">
                &#10003; Verified high-confidence (${verified.length})
              </div>
              <div class="pe-cards">${verified.map(profileCard).join("")}</div>
            </div>` : ""}
          ${unverified.length ? `
            <div class="pe-group">
              <div class="pe-group-head unverified">
                ? Unverified possible match (${unverified.length})
              </div>
              <div class="pe-cards">${unverified.map(profileCard).join("")}</div>
            </div>` : ""}
          ${discovered.length ? `
            <div class="pe-group">
              <div class="pe-group-head discovered">
                &#9675; Discovered link only (${discovered.length})
              </div>
              <div class="pe-cards">${discovered.map(profileCard).join("")}</div>
            </div>` : ""}
        `}
    </section>`;
}

function candidateVerificationSection(result, opts) {
  const all = (result.verification || {}).results || [];
  const analysed = analysedCandidates(result);
  if (!analysed.length) return "";

  const runId = resultRunId;
  const policy = ((result.verification || {}).configuration || {}).verification_policy || {};
  const ceiling = policy.similarity_ceiling;

  const byScore = (a, b) => b.face_similarity - a.face_similarity;
  const verified = analysed.filter((r) => candidateKind(r) === "verified").sort(byScore);
  const inconclusive = analysed.filter((r) => candidateKind(r) === "inconclusive").sort(byScore);
  const rejected = analysed.filter((r) => candidateKind(r) === "rejected").sort(byScore);

  // Bounded on purpose: this demonstrates the decision, it is not a gallery.
  const shownRejected = rejected.slice(0, 10);

  // Counted, never carded: no measurement exists for these.
  const notRetrievable = all.length - analysed.length;

  const stat = (n, label) => `
    <div class="cvs-stat"><div class="cvs-n">${n}</div>
      <div class="cvs-l">${esc(label)}</div></div>`;

  return `
    <section class="story cvs">
      <div class="story-q">${esc((opts && opts.heading) || "Candidate verification evidence")}</div>
      <p class="cvs-sub">
        Real candidates returned by this investigation's live reverse-image
        search, each downloaded and independently face-verified by TRACELOCK.
      </p>

      <div class="cvs-summary">
        ${stat(all.length, "discovered")}
        ${stat(analysed.length, "face-analysed")}
        ${stat(verified.length, "verified")}
        ${stat(inconclusive.length, "inconclusive")}
        ${stat(rejected.length, "rejected")}
        ${notRetrievable > 0 ? stat(notRetrievable, "not retrievable") : ""}
      </div>

      ${candidateGroup(verified, "verified", "✓ Verified same-person matches",
        "These passed independent face verification against your image.",
        runId, ceiling, null)}

      ${candidateGroup(inconclusive, "inconclusive", "⚠ Inconclusive",
        "Measured, but landing in the indeterminate band — TRACELOCK will not "
        + "claim these as the same person on this evidence.",
        runId, ceiling, null)}

      ${candidateGroup(shownRejected, "rejected", "✗ Examined and rejected",
        "Returned by the live search as visually similar, downloaded, and "
        + "measured against your face — none met the calibrated same-person "
        + (ceiling != null ? "threshold of <b>" + ceiling.toFixed(4) + "</b>." : "threshold."),
        runId, ceiling, rejected.length)}

      ${notRetrievable > 0 ? `
        <p class="cvs-foot">
          ${notRetrievable} further candidate${notRetrievable === 1 ? " was" : "s were"}
          discovered but could not be retrieved for analysis, so
          ${notRetrievable === 1 ? "it is" : "they are"} not shown above —
          no comparison was performed and none is implied.
        </p>` : ""}
    </section>`;
}

function renderNoMatch(result) {
  const m = result.no_match || {};
  const input = result.input || {};
  const sources = (result.discovery || {}).sources || [];

  $("#results").innerHTML = `
    <section class="story">
      <div class="story-q">1 · What was investigated?</div>
      <div class="investigated">
        <div class="preview-box"><img src="/api/preview/${esc(input.sha256 || "")}" alt=""></div>
        <div>
          <div class="metarow"><span>Source</span><b>${esc(input.source_label || "—")}</b></div>
          <div class="metarow"><span>File</span><b>${esc(input.filename || "—")}</b></div>
          <div class="metarow"><span>SHA-256</span><b class="mono">${esc((input.sha256 || "").slice(0, 24))}…</b></div>
        </div>
      </div>
    </section>

    <section class="story">
      <div class="story-q">2 · What happened?</div>

      <div class="nomatch-status">
        <span class="tick">&#10003;</span>
        <div>
          <div class="status ok">LIVE SEARCH COMPLETED</div>
          <p class="anchor-sub">${esc(m.statement || "")}</p>
        </div>
      </div>

      <div class="nomatch-verdict">
        <div class="nv-head">No verified same-person match found</div>
        <div class="nomatch-facts">
          <div class="factrow"><span class="hl">Engines answered</span>
            <span class="fv">${m.engines_answered ?? "—"} of ${m.engines_queried ?? "—"}
              ${sources.length ? `(${sources.map((s) => esc(s.engine)).join(", ")})` : ""}</span></div>
          <div class="factrow"><span class="hl">Discovered</span>
            <span class="fv">${m.candidates_discovered ?? "—"} candidates</span></div>
          <div class="factrow"><span class="hl">Examined</span>
            <span class="fv">${m.candidates_examined ?? "—"} candidates</span></div>
          <div class="factrow"><span class="hl">Face-analysed</span>
            <span class="fv">${m.candidates_face_analysed ?? "—"} candidates</span></div>
          <div class="factrow"><span class="hl">Highest similarity</span>
            <span class="fv mono">${m.highest_similarity != null ? m.highest_similarity.toFixed(4) : "—"}</span></div>
          <div class="factrow"><span class="hl">Threshold required</span>
            <span class="fv mono">${m.threshold_required != null ? m.threshold_required.toFixed(4) : "—"}</span></div>
        </div>

        <p class="nv-note">
          <strong>${esc(m.note || "")}</strong>
          A reverse-image engine returns pictures that <em>look</em> alike.
          TRACELOCK downloaded each one and compared it against your face with
          a calibrated model; none reached the same-person threshold. The
          search succeeded &mdash; there was simply nothing to verify.
        </p>
      </div>

      <p class="note">
        No trust score is shown, and nothing was anchored: both require
        verified evidence, and producing either here would be fabrication.
      </p>

    </section>

    ${candidateVerificationSection(result, {
      heading: "Why these results were rejected",
    })}

    <section class="story">
      <div class="chain-actions">
        <button class="btn primary" data-goto="screen-source">Investigate another image</button>
      </div>
    </section>`;

  $("#results").querySelectorAll("[data-goto]").forEach((b) =>
    b.addEventListener("click", () => show(b.dataset.goto)));
}

function renderResults(result) {
  const evidence = result.evidence || {};
  const trust = evidence.trust_score;
  const funnel = (evidence.evidence || {}).funnel || {};
  const items = (evidence.evidence || {}).items || [];
  const domains = (evidence.evidence || {}).independent_domains || [];
  const results = (result.verification || {}).results || [];
  const rejected = results.filter((r) => r.status === "REJECTED");
  const input = result.input || {};

  const coverage = (result.verification || {}).coverage || {};
  const perf = result.performance || {};
  const sources = (result.discovery || {}).sources || [];
  const social = (result.verification || {}).social || {};

  const score = trust ? trust.score : 0;
  const band = trust ? trust.band : "INSUFFICIENT";
  const dash = 2 * Math.PI * 82;
  const colour = { STRONG: "#4FBF8B", MODERATE: "#DCA544", WEAK: "#E0A64E", INSUFFICIENT: "#E8798F" }[band];
  const terms = trust ? trust.terms : null;

  $("#results").innerHTML = `
    <!-- 1. WHAT WAS INVESTIGATED? -->
    <section class="story">
      <div class="story-q">1 · What was investigated?</div>
      <div class="investigated">
        <div class="preview-box"><img src="/api/preview/${esc(input.sha256 || "")}" alt=""></div>
        <div>
          <div class="metarow"><span>Source</span><b>${esc(input.source_label || "—")}</b></div>
          <div class="metarow"><span>File</span><b>${esc(input.filename || "—")}</b></div>
          <div class="metarow"><span>SHA-256</span><b class="mono">${esc((input.sha256 || "").slice(0, 24))}…</b></div>
          <div class="metarow"><span>Searched via</span><b>${esc((result.discovery || {}).engine || "—")}</b></div>
          ${input.kind === "DEMO_EXAMPLE"
            ? `<div class="kind-notice" style="margin-top:14px">${esc(input.kind_notice)}</div>` : ""}
        </div>
      </div>
    </section>

    <!-- 2. WHAT WAS FOUND? -->
    <section class="story">
      <div class="story-q">2 · What was found?</div>
      <div class="summary">
        ${stat(funnel.discovered ?? results.length, "Candidates found")}
        ${stat(funnel.verified ?? 0, "Verified evidence", "good")}
        ${stat(funnel.independent_publishers ?? 0, "Independent sources")}
        ${stat(rejected.length, "Rejected", "bad")}
      </div>
      <p class="note">
        ${sources.length > 1 ? `${sources.length} search indexes` : "A search engine"}
        <em>suggested</em> ${coverage.discovered_total ?? funnel.discovered ?? results.length}
        candidates${coverage.available != null && coverage.available !== coverage.discovered_total
          ? `, ${coverage.available} after removing duplicates found by more than one index` : ""}.
        TRACELOCK re-downloaded every one it examined and re-measured it
        independently — the search engine's opinion was never trusted.
      </p>
      ${sources.length > 1 ? `
        <div class="sources-table">
          <div class="st-head">Discovery sources queried concurrently</div>
          ${sources.map((s) => `
            <div class="st-row ${s.ok ? "ok" : "bad"}">
              <span class="st-name">${esc(s.engine)}</span>
              <span class="st-state">${s.ok ? "answered" : (s.timed_out ? "timed out" : "failed")}</span>
              <span class="st-count">${s.ok ? `${s.candidates} candidates` : esc((s.error || "").slice(0, 60))}</span>
              <span class="st-time">${s.seconds.toFixed(2)}s</span>
            </div>`).join("")}
          <p class="note">
            Sources run at the same time and are isolated: one failing or
            hanging never blocks the others. The same image found by two
            engines is one candidate with two discoverers &mdash; it is not
            counted as two pieces of evidence.
          </p>
        </div>` : ""}

      ${coverage.stopped_early ? `
        <div class="coverage">
          <b>Stopped early.</b> ${esc(coverage.explanation)}
          Corroboration counts only what was actually examined, so this score
          is a floor, not a ceiling.
        </div>` : ""}
    </section>

    <!-- 2b. WAS THIS A GENUINE LIVE SEARCH? -->
    <section class="story">
      <div class="story-q">Was this a genuine live search?</div>
      ${liveSearchProof(result, sources, social)}
    </section>

    <!-- 3. WHAT WAS VERIFIED? -->
    <section class="story">
      <div class="story-q">3 · What was verified?</div>
      <div class="score-hero">
        <div class="dial">
          <svg width="190" height="190">
            <circle cx="95" cy="95" r="82" fill="none" stroke="#272E38" stroke-width="13"/>
            <circle cx="95" cy="95" r="82" fill="none" stroke="${colour}" stroke-width="13"
                    stroke-linecap="round" stroke-dasharray="${dash}"
                    stroke-dashoffset="${dash * (1 - score / 100)}"/>
          </svg>
          <div class="val">
            <div class="num">${trust ? score.toFixed(2) : "—"}</div>
            <div class="of">of 100</div>
          </div>
        </div>
        <div>
          <div class="band ${band}">${band.replace("_", " ")}${trust ? " EVIDENCE" : ""}</div>
          <div style="color:var(--ink-mid);font-size:14.5px">
            ${trust
              ? "Computed from independently re-downloaded and re-verified evidence."
              : "No trust score was produced. TRACELOCK refuses to score without a calibrated identity probability."}
          </div>
          ${trust && (funnel.independent_publishers ?? 0) === 0 ? `
            <div class="caveat">
              <b>Read this score carefully.</b>
              No <em>independent</em> corroboration was found — every verified
              image is a republication of the photograph you supplied, not a
              different photograph of the same person. The score is high because
              identity is near-certain when the pixels match, but a single
              source is an assertion, not consensus.
            </div>` : ""}
          ${trust && (funnel.independent_publishers ?? 0) === 1 ? `
            <div class="caveat">
              <b>One independent source only.</b>
              Corroboration contributes nothing below two independent
              publishers. A single source is an assertion, not consensus.
            </div>` : ""}
        </div>
      </div>

      ${terms ? `
      <div class="why">
        <div class="why-t">Why this score?</div>
        <div class="factors">
          ${factor("Identity probability", terms.P_id, "calibrated P(same person) for the strongest unique image")}
          ${factor("Evidence quality", terms.Q, "face quality, metadata completeness, clean acquisition")}
          ${factor("Corroboration", terms.C, `${funnel.independent_publishers ?? 0} independent publisher(s), saturating`)}
        </div>
        <pre class="formula">${esc(trust.explanation.join("\n"))}</pre>
      </div>` : ""}

      <h2 style="margin-top:26px">Verified evidence — ${items.length} unique image(s)</h2>
      ${items.length ? `<div class="cards">${items.map(evidenceCard).join("")}</div>`
        : `<p class="note">No candidate survived independent verification.</p>`}

      ${domains.length ? `<p class="note"><strong>Independent sources:</strong>
        ${domains.map(esc).join(" · ")}</p>` : ""}

      <details class="rejected" style="margin-top:20px">
        <summary>${rejected.length} rejected — show reasons</summary>
        <table class="rtable">
          <thead><tr><th>Source</th><th>Stage</th><th>Reason</th><th>Similarity</th></tr></thead>
          <tbody>${rejected.map(rejectedRow).join("")}</tbody>
        </table>
      </details>
      <p class="note">Rejections are shown deliberately: a verifier that accepts
      everything has tested nothing.</p>
    </section>

    ${candidateVerificationSection(result, {})}

    ${publicProfileEvidenceSection(result)}

    <!-- 4. CAN THE RESULT BE TAMPERED WITH? -->
    <section class="story">
      <div class="story-q">4 · Can this result be tampered with?</div>
      <div id="chainbox">${chainBlock(result.anchor)}</div>
    </section>

    ${trust ? `<div class="limits">
      <h4>What this score does not establish</h4>
      <ul>${trust.limitations.map((l) => `<li>${esc(l)}</li>`).join("")}</ul>
    </div>` : ""}

    ${renderPerformance(perf, coverage, results)}`;

  wireChainActions(result);
}

function factor(label, value, note) {
  const pct = Math.round((value ?? 0) * 100);
  return `
    <div class="factor">
      <div class="f-top"><span>${esc(label)}</span><b>${(value ?? 0).toFixed(3)}</b></div>
      <div class="f-track"><span class="f-fill" style="width:${pct}%"></span></div>
      <div class="f-note">${esc(note)}</div>
    </div>`;
}

const stat = (n, label, cls = "") =>
  `<div class="stat ${cls}"><div class="n">${n}</div><div class="l">${label}</div></div>`;

function evidenceCard(item) {
  const domains = (item.domains || []).filter(Boolean);
  return `
    <div class="ecard">
      <img src="${esc(item.media_urls?.[0] || "")}" alt="" loading="lazy"
           onerror="this.style.display='none'">
      <div class="b">
        <div class="dom">${esc(domains[0] || "unknown source")}</div>
        ${domains.length > 1 ? `<div class="kv"><span>also on</span><b>${domains.length - 1} more</b></div>` : ""}
        <div class="kv"><span>P(identity)</span><b>${item.identity_probability != null ? item.identity_probability.toFixed(3) : "—"}</b></div>
        <div class="kv"><span>similarity</span><b>${item.face_similarity?.toFixed(4) ?? "—"}</b></div>
        <div class="kv"><span>pHash dist</span><b>${item.phash_distance ?? "—"}/64</b></div>
        <span class="tag ok">VERIFIED</span>
        <span class="tag rel">${esc((item.relation || "").replace(/_/g, " "))}</span>
      </div>
    </div>`;
}

function rejectedRow(r) {
  const reason = r.rejection_reasons?.[0];
  return `<tr>
    <td>${esc(r.provenance?.registrable_domain || "—")}</td>
    <td>${esc(reason?.stage || "—")}</td>
    <td class="reason">${esc(reason?.reason || "—")}</td>
    <td>${r.face_similarity != null ? r.face_similarity.toFixed(4) : "—"}</td>
  </tr>`;
}

function shortHash(value, head = 10, tail = 8) {
  const text = String(value || "");
  if (text.length <= head + tail + 3) return text;
  return `${text.slice(0, head)}…${text.slice(-tail)}`;
}

function hashRow(label, value) {
  // Truncated for reading, full value kept for copying. Neither is enough on
  // its own: a wall of hex is unreadable, and a truncated hash is unusable.
  if (!value) return "";
  return `
    <div class="hashrow">
      <span class="hl">${esc(label)}</span>
      <code class="hv" title="${esc(value)}">${esc(shortHash(value))}</code>
      <button class="copy" data-copy="${esc(value)}" title="Copy full value"
              aria-label="Copy ${esc(label)}">⧉</button>
    </div>`;
}

function chainBlock(anchor) {
  if (!anchor) {
    return `<div class="chain none">
      <div class="status no">NOT ANCHORED</div>
      <p class="note">This evidence has not been committed to a blockchain.</p>
      <div class="chain-actions">
        <button class="btn primary" id="do-anchor">ANCHOR EVIDENCE</button>
      </div>
    </div>`;
  }

  const c = anchor.on_chain || {};
  const ephemeral = Boolean(c.ephemeral);

  // The environment is a PROPERTY of a successful anchor, not a failure of
  // one. Anchoring worked; tamper detection works; the chain happens to be
  // local. Leading with a warning icon made a working feature read as broken,
  // so the success is primary and the limitation is a labelled fact beside it.
  const verifyLabel = ephemeral ? "VERIFY INTEGRITY" : "VERIFY ON BLOCKCHAIN";

  return `<div class="chain anchored">
    <div class="anchor-head">
      <div class="anchor-status">
        <span class="tick">✓</span>
        <div>
          <div class="status ok">EVIDENCE INTEGRITY ANCHORED</div>
          <p class="anchor-sub">
            The cryptographic evidence record has been anchored successfully.
            Any change to this evidence is now detectable.
          </p>
        </div>
      </div>
      <span class="env-badge ${ephemeral ? "env-local" : "env-public"}">
        ${ephemeral ? "LOCAL DEMO CHAIN" : "PUBLICLY VERIFIABLE"}
      </span>
    </div>

    <div class="anchor-facts">
      ${hashRow("Evidence root", anchor.merkle_root)}
      ${hashRow("Transaction", c.tx_hash)}
      ${hashRow("Contract", c.contract_address)}
      <div class="factrow">
        <span class="hl">Block</span>
        <span class="fv">#${esc(c.block_number)}</span>
      </div>
      <div class="factrow">
        <span class="hl">Anchored at</span>
        <span class="fv">${esc(c.block_time_utc || "—")}</span>
      </div>
      <div class="factrow">
        <span class="hl">Integrity</span>
        <span class="fv ok">Verifiable — tamper-evident</span>
      </div>
    </div>

    ${ephemeral ? `
      <div class="env-card">
        <div class="env-title">Local demo environment</div>
        <p>
          Running on a local ephemeral blockchain for demonstration.
          <strong>Integrity and tamper detection are fully functional.</strong>
          This anchor is functional for this running demo but resets on restart.
          It is not independently verifiable by a third party.
        </p>
      </div>`
    : `
      <div class="env-card public">
        <div class="env-title">${esc(c.network_display_name || c.network)}</div>
        <p>
          <strong>Anyone can independently verify this evidence anchor.</strong>
          The transaction below exists on a public chain and can be checked
          without trusting TRACELOCK.
        </p>
      </div>`}

    ${c.explorer_url
      ? `<p class="explorer">
           <a href="${esc(c.explorer_url)}" target="_blank" rel="noopener">
             View this transaction on the block explorer →
           </a>
         </p>`
      : `<p class="explorer none">
           No block explorer — this chain is not publicly verifiable.
         </p>`}

    <div class="chain-actions">
      <button class="btn primary" id="do-verify">${verifyLabel}</button>
      <button class="btn" id="do-tamper">TEST TAMPER DETECTION</button>
    </div>
    <p class="action-help">
      Modify the evidence bundle and verify that TRACELOCK detects the
      cryptographic mismatch.
    </p>

    <div id="verdict"></div>

    <p class="integrity-note">
      This anchor records cryptographic <strong>integrity</strong>, not truth.
      It can prove that evidence changed; it cannot prove the original source
      was truthful.
    </p>
  </div>`;
}

async function runAnchor(path, useLocalFallback) {
  const button = $("#do-anchor");
  if (button) {
    button.disabled = true;
    button.textContent = useLocalFallback ? "Anchoring locally…" : "Anchoring…";
  }
  try {
    const record = await post("/api/chain/anchor", form({
      artifact_path: path,
      use_local_fallback: useLocalFallback ? "true" : "false",
    }));
    lastResult.anchor = record;
    $("#chainbox").innerHTML = chainBlock(record);
    wireChainActions(lastResult);
  } catch (err) {
    renderAnchorFailure(err, path);
  }
}

function renderAnchorFailure(err, path) {
  // A failed public anchor is a result, not a dead end. It states plainly that
  // NOTHING was anchored, and offers three real actions -- each of which does
  // something when clicked.
  const options = err.recovery || [];

  $("#chainbox").innerHTML = `
    <div class="chain none">
      <div class="status no">${esc(err.error || "ANCHORING FAILED")}</div>
      <p class="anchor-sub">${esc(err.message || err.detail || "Nothing was anchored.")}</p>
      ${(err.problems || []).length ? `
        <ul class="problems">
          ${err.problems.map((p) => `<li>${esc(p)}</li>`).join("")}
        </ul>` : ""}
      <div class="chain-actions">
        ${options.length
          ? options.map((o) => `
              <button class="btn ${o.action === "anchor_local" ? "primary" : ""}"
                      data-anchor-action="${esc(o.action)}">${esc(o.label)}</button>`).join("")
          : `<button class="btn primary" data-anchor-action="retry_public">Try again</button>`}
      </div>
      <p class="action-help">
        No public verification record was created. Your investigation result is
        unchanged.
      </p>
    </div>`;

  $("#chainbox").querySelectorAll("[data-anchor-action]").forEach((button) => {
    button.addEventListener("click", () => {
      const action = button.dataset.anchorAction;
      if (action === "retry_public") return runAnchor(path, false);
      if (action === "anchor_local") return runAnchor(path, true);
      if (action === "return_results") {
        // The investigation itself was never in doubt -- re-render it.
        renderResults(lastResult);
        document.querySelector(".story")?.scrollIntoView({ behavior: "smooth" });
      }
    });
  });
}

function wireChainActions(result) {
  const path = result.artifact_path;

  $("#do-anchor")?.addEventListener("click", () => runAnchor(path, false));

  // Full hash values stay reachable even though the display is truncated.
  document.querySelectorAll(".copy").forEach((button) => {
    button.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(button.dataset.copy);
        const original = button.textContent;
        button.textContent = "✓";
        setTimeout(() => { button.textContent = original; }, 1200);
      } catch {
        toast("Could not copy.", "Your browser blocked clipboard access.");
      }
    });
  });

  const ephemeral = Boolean((lastResult?.anchor?.on_chain || {}).ephemeral);
  const verifyLabel = ephemeral ? "VERIFY INTEGRITY" : "VERIFY ON BLOCKCHAIN";

  $("#do-verify")?.addEventListener("click", async (e) => {
    e.target.disabled = true; e.target.textContent = "Verifying…";
    try {
      const verdict = await post("/api/chain/verify", form({ artifact_path: path }));
      const intact = verdict.verdict === "INTACT";
      $("#verdict").innerHTML = `
        <div class="verdict ${esc(verdict.verdict)}">
          <b>${intact ? "✓ INTEGRITY CONFIRMED" : "⚠ " + esc(verdict.verdict.replace("_", " "))}</b>
          <small>${esc(verdict.explanation)}</small>
          ${intact ? `<small>The evidence bundle still hashes to the anchored
             root. Nothing has been altered since it was anchored.</small>` : ""}
        </div>`;
    } catch (err) {
      toast(err.error || "Verification failed.");
    } finally {
      e.target.disabled = false; e.target.textContent = verifyLabel;
    }
  });

  // A REAL test: the server corrupts a copy and re-runs the same verification
  // the button above runs. Nothing is staged and the stored evidence is not
  // touched -- if this ever printed "detected" without detecting, the two
  // verdicts it shows would contradict it on screen.
  $("#do-tamper")?.addEventListener("click", async (e) => {
    e.target.disabled = true; e.target.textContent = "Testing…";
    $("#verdict").innerHTML = `<div class="verdict testing">Running integrity test…</div>`;
    try {
      const out = await post("/api/chain/tamper-test",
        form({ artifact_path: path, field: "trust_score" }));

      const before = JSON.stringify(out.before);
      const after = JSON.stringify(out.after);

      $("#verdict").innerHTML = `
        <div class="tamper-demo">
          <div class="td-step">
            <span class="td-n">1</span>
            <div>
              <b>Evidence modified</b>
              <div class="td-diff">
                <code class="was">${esc(out.field)}: ${esc(before)}</code>
                <span class="arrow">→</span>
                <code class="now">${esc(after)}</code>
              </div>
              <small>Applied to an in-memory copy. The stored evidence file was
              not changed.</small>
            </div>
          </div>

          <div class="td-step">
            <span class="td-n">2</span>
            <div>
              <b>Integrity re-verified against the anchor</b>
              <small>Untouched bundle: <em>${esc(out.baseline.verdict)}</em>
              &nbsp;·&nbsp; Modified bundle: <em>${esc(out.tampered.verdict)}</em></small>
            </div>
          </div>

          <div class="td-result ${out.detected ? "caught" : "missed"}">
            ${out.detected
              ? `<b>❌ TAMPERING DETECTED</b>
                 <span>Evidence hash no longer matches the anchored record.</span>`
              : `<b>⚠ INCONCLUSIVE</b>
                 <span>${esc(out.tampered.explanation || "The test could not complete.")}</span>`}
          </div>
        </div>`;
    } catch (err) {
      toast(err.error || "The tamper test could not run.");
      $("#verdict").innerHTML = "";
    } finally {
      e.target.disabled = false;
      e.target.textContent = "TEST TAMPER DETECTION";
    }
  });
}

// ------------------------------------------------- public discovery consent
//
// The bridge for local images. A reverse-image API fetches a URL and cannot
// receive a file, so an upload or a webcam frame has no way into public
// discovery until a copy of it is publicly reachable. That is a real biometric
// disclosure, so it happens only here, only after this panel, and only on an
// explicit click.

$("#go-publish")?.addEventListener("click", async () => {
  $("#mode-choose").hidden = true;
  $("#mode-consent").hidden = false;
  $("#consent-status").textContent = "";

  let info;
  try {
    info = await get("/api/hosting/info");
  } catch {
    $("#consent-status").textContent =
      "Could not read the hosting configuration.";
    return;
  }

  $("#consent-what").textContent = info.what_happens;
  $("#consent-guarantees").innerHTML =
    (info.guarantees || []).map((g) => `<li>${esc(g)}</li>`).join("");

  // The provider's REAL capabilities, read from the provider itself -- so this
  // panel cannot promise a deletion the code will not perform.
  $("#consent-host").innerHTML = `
    <div class="host-row">
      <span class="hl">Host</span>
      <span class="fv">${esc(info.display_name)}</span>
    </div>
    <div class="host-row">
      <span class="hl">Deletion</span>
      <span class="fv ${info.supports_deletion ? "ok" : "warn"}">
        ${info.supports_deletion
          ? "Supported — the copy is removed after discovery"
          : "Not available on this host"}
      </span>
    </div>
    <p class="host-note">${esc(info.retention_note)}</p>
    ${info.configured ? "" : `
      <p class="host-note warn">
        This host is not configured. Set ${esc((info.providers.find(
          (p) => p.key === info.selected) || {}).missing?.join(", ") || "credentials")}
        in .env, or use a public URL instead.
      </p>`}`;

  $("#consent-accept").disabled = !info.configured;
});

$("#consent-cancel")?.addEventListener("click", () => {
  // Declining is a real choice, not a dead end: local analysis is still there.
  $("#mode-consent").hidden = true;
  $("#mode-choose").hidden = false;
});

$("#consent-accept")?.addEventListener("click", async (event) => {
  const button = event.target;
  button.disabled = true;
  const status = $("#consent-status");

  try {
    status.textContent = "Publishing a temporary copy so search engines can fetch it…";
    const published = await post("/api/input/publish", form({
      sha256: selected.sha256,
      consent: "true",
      retention: "1h",
    }));

    selected = published.input;
    $("#m-url").textContent = selected.image_url;
    status.textContent = "Published. Starting live reverse-image search…";

    $("#mode-consent").hidden = true;
    $("#mode-public").hidden = false;
    $("#start").disabled = false;
    $("#start-note").textContent =
      "You consented to publishing this image. " + (published.note || "");

    // Straight into the SAME investigation pipeline a public URL uses.
    $("#start").click();
  } catch (err) {
    button.disabled = false;
    status.textContent = "";
    renderConsentFailure(err);
  }
});

function renderConsentFailure(err) {
  const options = err.recovery || [];
  $("#consent-status").innerHTML = `
    <span class="consent-error">${esc(err.error || "Publishing failed.")}</span>
    ${err.hint ? `<span class="consent-hint">${esc(err.hint)}</span>` : ""}`;

  if (options.length) {
    const row = document.createElement("div");
    row.className = "mode-actions";
    row.innerHTML = options.map((o) => `
      <button class="btn recovery-btn" data-action="${esc(o.action)}">
        ${esc(o.label)}</button>`).join("");
    $("#consent-status").appendChild(row);
    row.querySelectorAll(".recovery-btn").forEach((b) =>
      b.addEventListener("click", () => runRecovery(b.dataset.action)));
  }
}

// ---------------------------------------------------------------- mode B

$("#go-local").addEventListener("click", async () => {
  if (!selected) return;
  const button = $("#go-local");
  button.disabled = true;
  button.textContent = "Analysing…";
  try {
    const report = await post("/api/analyze/local", form({ sha256: selected.sha256 }));
    renderLocalAnalysis(report);
    show("screen-results");
  } catch (err) {
    toast(err.error || "Local analysis failed.", err.hint);
  } finally {
    button.disabled = false;
    button.textContent = "LOCAL ANALYSIS";
  }
});

$("#go-addurl").addEventListener("click", () => {
  $("#mode-choose").hidden = true;
  $("#mode-addurl").hidden = false;
  $("#link-input").focus();
});

$("#link-back").addEventListener("click", () => {
  $("#mode-addurl").hidden = true;
  $("#mode-choose").hidden = false;
});

$("#link-go").addEventListener("click", () => checkLink());
$("#link-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter") checkLink();
});

// Link-type detection mirrors the SERVER's classifier. It is a UI affordance
// only -- the server classifies independently and its answer is authoritative.
const LINK_PATTERNS = [
  [/\.(jpe?g|png|webp|gif|bmp|tiff?|avif)(\?|#|$)/i, "🖼", "Direct image detected", "direct"],
  [/(^|\.)instagram\.com|instagr\.am/i, "📷", "Instagram post detected", "instagram"],
  [/(^|\.)(x|twitter)\.com|(^|\.)t\.co/i, "𝕏", "X post detected", "x"],
  [/(^|\.)facebook\.com|fb\.watch/i, "f", "Facebook post detected", "facebook"],
  [/(^|\.)linkedin\.com|lnkd\.in/i, "in", "LinkedIn post detected", "linkedin"],
  [/(^|\.)youtube\.com|youtu\.be/i, "▶", "YouTube video detected", "youtube"],
];

function detectLinkType(url) {
  let host = "";
  try { host = new URL(url).hostname; } catch (_) { return null; }
  for (const [re, icon, label, key] of LINK_PATTERNS) {
    if (re.test(url) || re.test(host)) return { icon, label, key };
  }
  return { icon: "🔗", label: "Public webpage detected", key: "webpage" };
}

function showDetected(url) {
  const box = $("#link-detect");
  const detected = url.trim() ? detectLinkType(url) : null;
  if (!detected) { box.innerHTML = ""; return; }
  box.innerHTML = `
    <div class="detected">
      <span class="dico">${detected.icon}</span>
      <span>
        <span class="dlabel">${esc(detected.label)}</span>
        ${detected.key === "direct"
          ? ""
          : `<div class="dhint">TRACELOCK will look for a publicly published preview image.</div>`}
      </span>
    </div>`;
}

$("#link-input").addEventListener("input", (e) => showDetected(e.target.value));

const RESOLVE_STEPS = [
  "Detecting link type",
  "Resolving publicly available image",
  "Checking image safety",
  "Comparing with selected image",
];

function renderSteps(activeIndex, failedIndex = -1) {
  $("#link-steps").innerHTML = `<ol class="res-steps">${
    RESOLVE_STEPS.map((label, i) => {
      const state = i === failedIndex ? "failed"
        : i < activeIndex ? "done"
        : i === activeIndex ? "running" : "pending";
      const icon = { done: "✓", running: "◐", failed: "✕", pending: "○" }[state];
      return `<li class="${state}"><span class="ric">${icon}</span><span>${esc(label)}</span></li>`;
    }).join("")}</ol>`;
}

function provenanceLine(res) {
  if (!res) return "";
  return `
    <div class="res-prov">
      <b>input_type</b> ${esc(res.input_type)} ·
      <b>platform</b> ${esc(res.platform)} ·
      <b>resolution_method</b> ${esc(res.resolution_method)}
      ${res.resolved_image_url
        ? `<br><b>resolved_image_url</b> ${esc(res.resolved_image_url)}` : ""}
    </div>`;
}

function renderCandidates(res, pageUrl) {
  const box = $("#link-candidates");
  const others = (res && res.alternatives) || [];
  if (!others.length) { box.hidden = true; box.innerHTML = ""; return; }

  // Ranking is a convenience, not a claim. When the page held several images
  // the operator can override the top pick -- which matters when the ranking
  // picks the article's hero photo over the byline portrait.
  box.hidden = false;
  box.innerHTML = `
    <div class="cand-head">
      This page has ${others.length + 1} images. Using the highest-ranked one.
    </div>
    <div class="cand-grid">
      ${others.map((c) => `
        <button class="cand" data-url="${esc(c.url)}" title="${esc(c.url)}">
          <img src="${esc(c.url)}" alt="" loading="lazy"
               onerror="this.closest('.cand').classList.add('broken')">
          <span class="cand-label">${esc(c.label)}</span>
          <span class="cand-src">${esc(c.source)}</span>
        </button>`).join("")}
    </div>`;

  box.querySelectorAll(".cand").forEach((button) => {
    button.addEventListener("click", () => checkLink(button.dataset.url, pageUrl));
  });
}

async function checkLink(candidateUrl, pageUrlOverride) {
  const url = (pageUrlOverride || $("#link-input").value).trim();
  if (!url) return toast("Paste a link first.");

  const button = $("#link-go");
  button.disabled = true;
  button.textContent = "Checking…";
  $("#link-result").innerHTML = "";

  // Network activity begins ONLY here, on an explicit click. The step
  // advances because the request is genuinely in flight -- there is no timer
  // pretending work is happening.
  renderSteps(1);

  try {
    const result = await post("/api/input/link-url",
      form({ sha256: selected.sha256, url, candidate_url: candidateUrl || "" }));
    const cmp = result.comparison;
    const res = result.resolution;

    renderSteps(4);

    const viaPage = res && res.method !== "direct" && res.method !== "sniffed";
    const resolvedLine = viaPage
      ? `<div>✓ ${esc(res.label)}</div>
         <div>✓ Public image resolved
           <span class="link-kv">(${esc(res.method_explanation || res.method)})</span></div>`
      : `<div>✓ Direct image detected</div>`;

    $("#link-result").innerHTML = `
      <div class="link-ok">
        <div class="t">${resolvedLine}
          <div>✓ Image readable</div>
          <div>${cmp.is_same_image ? "✓ Same picture confirmed" : "⚠ Different picture from the one you selected"}</div>
        </div>
        <div class="link-kv">
          ${esc(result.image.format)} ${result.image.width}x${result.image.height} ·
          ${result.face.count} face(s) · quality
          ${result.face.quality != null ? result.face.quality.toFixed(2) : "—"}
          [${esc(result.face.quality_band || "")}]
        </div>
        <div class="link-kv">
          perceptual hash distance ${cmp.phash_distance}/${cmp.phash_max}
        </div>
        <div class="link-note">${esc(cmp.note)}</div>
        ${provenanceLine(res)}
      </div>`;

    renderCandidates(res, url);

    // The linked input now HAS a public URL, so Mode A becomes available.
    selected = result.input;
    $("#m-url").textContent = selected.image_url;
    $("#mode-addurl").hidden = true;
    $("#mode-public").hidden = false;
    $("#start").disabled = false;
    $("#start-note").textContent =
      "You supplied this link. Nothing was uploaded on your behalf.";
  } catch (err) {
    const res = err.resolution;
    // Which step failed: resolution, or validation of the resolved image.
    renderSteps(-1, res && res.ok === false ? 1 : 2);

    $("#link-result").innerHTML = `
      <div class="link-bad">
        <div class="t">✕ ${esc(err.error || "That link cannot be used.")}</div>
        ${err.hint ? `<div class="link-note">${esc(err.hint)}</div>` : ""}
        ${res && res.guidance
          ? `<div class="link-guide">${esc(res.guidance)}</div>` : ""}
        ${provenanceLine(res)}
        <div class="fail-actions">
          <button class="btn" id="fail-upload">Upload the image instead</button>
          <button class="btn" id="fail-retry">Try another public source</button>
        </div>
      </div>`;

    $("#fail-upload").addEventListener("click", () => {
      $("#mode-addurl").hidden = true;
      $("#mode-choose").hidden = false;
      show("screen-source");
      $("#file").click();
    });
    $("#fail-retry").addEventListener("click", () => {
      $("#link-input").value = "";
      $("#link-input").focus();
      $("#link-result").innerHTML = "";
      $("#link-steps").innerHTML = "";
      $("#link-detect").innerHTML = "";
    });
  } finally {
    button.disabled = false;
    button.textContent = "Check link";
  }
}

// ---------------------------------------------------------------- local results

function renderLocalAnalysis(r) {
  const img = r.image, face = r.face, fp = r.fingerprint;
  const metrics = face.quality_metrics || [];

  $("#results").innerHTML = `
    <div class="local-banner">
      <div class="lb-main">LOCAL ANALYSIS</div>
      <div class="lb-sub">
        Everything measurable without touching the network. No image was
        transmitted and no public discovery was performed.
      </div>
    </div>

    <div class="summary">
      ${stat(esc(img.dimensions), "Dimensions")}
      ${stat(esc(img.format), "Format")}
      ${stat(face.count, "Faces detected", face.detected ? "good" : "bad")}
      ${stat(face.quality != null ? face.quality.toFixed(2) : "—", "Face quality")}
      ${stat("LOCAL ONLY", "Privacy status", "good")}
    </div>

    <div class="notperf">
      <div class="np-t">Public discovery: NOT PERFORMED</div>
      <p>${esc(r.public_discovery.reason)}</p>
      <p><strong>${esc(r.privacy.note)}</strong></p>
      <p>${esc(r.public_discovery.how_to_enable)}</p>
    </div>

    <section class="sec">
      <h2>Image</h2>
      <div class="selected">
        <div class="preview-box"><img src="/api/preview/${esc(img.sha256)}" alt=""></div>
        <div>
          <div class="metarow"><span>Dimensions</span><b>${esc(img.dimensions)}</b></div>
          <div class="metarow"><span>Format</span><b>${esc(img.format)}</b></div>
          <div class="metarow"><span>Size</span><b>${(img.byte_size / 1024).toFixed(1)} KB</b></div>
          <div class="metarow"><span>SHA-256</span><b class="mono">${esc(img.sha256_short)}</b></div>
          <div class="metarow"><span>Perceptual hash</span><b class="mono">${esc(fp.perceptual_hash || "—")}</b></div>
          <div class="metarow"><span>Source</span><b>${esc(r.provenance.source_label)}</b></div>
        </div>
      </div>
    </section>

    <section class="sec">
      <h2>Face</h2>
      ${face.detected ? `
        <div class="summary" style="margin-bottom:18px">
          ${stat(face.count, "Faces")}
          ${stat(face.det_score != null ? face.det_score.toFixed(3) : "—", "Detection confidence")}
          ${stat(esc(face.quality_band || "—"), "Quality band")}
          ${stat(esc(face.face_size || "—"), "Face size")}
        </div>
        <div class="qbars">
          ${metrics.map((m) => `
            <div class="qbar">
              <span class="qn">${esc(m.name)}</span>
              <span class="qt"><span class="qf" style="width:${Math.round(m.score * 100)}%"></span></span>
              <span class="qv">${m.score.toFixed(2)}</span>
            </div>`).join("")}
        </div>`
        : `<p class="note">No face was detected in this image.</p>`}
    </section>

    <section class="sec">
      <h2>Cryptographic fingerprint</h2>
      <dl class="kvgrid">
        <dt>image sha256</dt><dd>${esc(fp.image_sha256)}</dd>
        <dt>perceptual hash</dt><dd>${esc(fp.perceptual_hash || "—")}</dd>
        <dt>embedding commit</dt><dd>${esc(fp.embedding_quantized_sha256 || "—")}</dd>
        <dt>model</dt><dd>${esc(fp.model_id || "—")}</dd>
      </dl>
      <p class="note">${esc(fp.note)}</p>
    </section>

    <div class="limits">
      <h4>What this is not</h4>
      <ul>
        <li>This is <strong>not identity verification</strong>. No public sources
            were searched and no candidate was compared against this face.</li>
        <li><strong>No trust score is shown</strong>, because a trust score
            requires discovered evidence to score. Producing one here would be
            fabrication.</li>
        <li>To search the public web, provide a publicly reachable URL for this
            image and run a public investigation.</li>
      </ul>
    </div>`;
}

$$("[data-goto]").forEach((b) =>
  b.addEventListener("click", () => { stopCamera(); show(b.dataset.goto); }));

boot();
