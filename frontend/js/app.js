/* ============================================================
   PatchDentalNet DSS — Frontend Logic
   The backend is served from the same origin, so requests use a
   relative path (/api/...) — no separate address is needed.
   ============================================================ */

const API_HEALTH = "/api/health";
const API_PREDICT = "/api/predict";
const API_SHAP_REPORT = "/api/shap-report";

// SHAP feature group order/colors - must match svm_pipeline.FEATURE_GROUPS exactly.
const SHAP_GROUP_ORDER = [
  "Attention Context (Attn Mean)",
  "Patch Std",
  "Patch Max",
  "Attention Weights",
  "Softmax Scores",
];
const SHAP_GROUP_COLORS = {
  "Attention Context (Attn Mean)": "#2a78d6",
  "Patch Std": "#1baf7a",
  "Patch Max": "#7c5cbf",
  "Attention Weights": "#eb6834",
  "Softmax Scores": "#eda100",
};

// Class order must match the backend (predictor.py CLASSES) exactly.
// Colors match the --series-1..4 variables in css/style.css.
const CLASS_ORDER = ["No Lesion", "Radiolucent", "Radiopaque", "Mixed"];
const CLASS_COLORS = {
  "No Lesion": "#2a78d6",
  "Radiolucent": "#eb6834",
  "Radiopaque": "#1baf7a",
  "Mixed": "#eda100",
};

const el = (id) => document.getElementById(id);

const dropzone = el("dropzone");
const fileInput = el("fileInput");
const dropzoneEmpty = el("dropzoneEmpty");
const previewImg = el("previewImg");
const clearImageBtn = el("clearImageBtn");
const runBtn = el("runBtn");
const modelBadge = el("modelBadge");

const resultsEmpty = el("resultsEmpty");
const resultsContent = el("resultsContent");
const loadingOverlay = el("loadingOverlay");
const errorBanner = el("errorBanner");

const caseIdInput = el("caseId");
const ageInput = el("age");
const genderInput = el("gender");
const complaintInput = el("complaint");
const notesInput = el("notes");

let selectedFile = null;
let lastResult = null;

// ---------------------------------------------------------------
// Startup
// ---------------------------------------------------------------

function initCaseId() {
  const now = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  const id = `CASE-${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}`;
  caseIdInput.value = id;
}

async function checkHealth() {
  try {
    const res = await fetch(API_HEALTH);
    if (!res.ok) throw new Error("health check failed");
    const data = await res.json();
    modelBadge.className = "badge badge-ok";
    const modeLabel = data.model_mode === "real" ? "Real Model" : "Mock Mode (test data)";
    modelBadge.innerHTML = `<span class="badge-dot"></span> Backend connected · ${modeLabel}`;
  } catch (err) {
    modelBadge.className = "badge badge-error";
    modelBadge.innerHTML = `<span class="badge-dot"></span> Cannot reach backend`;
  }
}

initCaseId();
checkHealth();

// ---------------------------------------------------------------
// Image selection (click / drag & drop)
// ---------------------------------------------------------------

fileInput.addEventListener("change", (e) => {
  if (e.target.files && e.target.files[0]) handleFile(e.target.files[0]);
});

["dragenter", "dragover"].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
});
["dragleave", "drop"].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  });
});
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files && e.dataTransfer.files[0];
  if (file) handleFile(file);
});

function handleFile(file) {
  if (!file.type.startsWith("image/")) {
    showError("Please select a PNG or JPG image.");
    return;
  }
  selectedFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewImg.classList.remove("hidden");
  dropzoneEmpty.classList.add("hidden");
  clearImageBtn.classList.remove("hidden");
  runBtn.disabled = false;
  hideError();
}

clearImageBtn.addEventListener("click", (e) => {
  e.preventDefault();
  selectedFile = null;
  fileInput.value = "";
  previewImg.src = "";
  previewImg.classList.add("hidden");
  dropzoneEmpty.classList.remove("hidden");
  clearImageBtn.classList.add("hidden");
  runBtn.disabled = true;
});

// ---------------------------------------------------------------
// Run analysis
// ---------------------------------------------------------------

runBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  hideError();
  setLoading(true);

  const formData = new FormData();
  formData.append("image", selectedFile);
  formData.append("case_id", caseIdInput.value);
  formData.append("age", ageInput.value);
  formData.append("gender", genderInput.value);
  formData.append("complaint", complaintInput.value);
  formData.append("notes", notesInput.value);

  let res;
  try {
    res = await fetch(API_PREDICT, { method: "POST", body: formData });
  } catch (err) {
    showError(`Analysis failed: ${err.message}. Make sure the backend (main.py) is running.`);
    setLoading(false);
    return;
  }
  try {
    if (!res.ok) {
      // 400/422: the backend rejected the file (unreadable or not a panoramic radiograph)
      const message = await readErrorDetail(res);
      lastResult = null;
      resultsContent.classList.add("hidden");
      resultsEmpty.classList.remove("hidden");
      showError(message);
      return;
    }
    const result = await res.json();
    lastResult = result;
    renderResults(result);
  } catch (err) {
    showError(`Analysis failed: ${err.message}`);
  } finally {
    setLoading(false);
  }
});

function setLoading(isLoading) {
  loadingOverlay.classList.toggle("hidden", !isLoading);
  runBtn.disabled = isLoading || !selectedFile;
}

async function readErrorDetail(res) {
  try {
    const body = await res.json();
    if (typeof body.detail === "string") return body.detail;
  } catch (_) {
    // non-JSON error body
  }
  return `Server error (${res.status})`;
}

function showError(msg) {
  errorBanner.textContent = msg;
  errorBanner.classList.remove("hidden");
}
function hideError() {
  errorBanner.classList.add("hidden");
}

// ---------------------------------------------------------------
// Render results
// ---------------------------------------------------------------

function renderResults(result) {
  resultsEmpty.classList.add("hidden");
  resultsContent.classList.remove("hidden");

  el("statPrediction").textContent = result.prediction;
  el("statConfidence").textContent = `${(result.confidence * 100).toFixed(1)}%`;
  el("statTriageText").textContent = result.triage;

  const triageDot = el("triageDot");
  triageDot.className = "triage-dot";
  if (result.triage_level) triageDot.classList.add(`level-${result.triage_level}`);

  renderProbChart(result.probabilities);
  renderFindings(result.findings);
  el("suggestionText").textContent = result.suggestion;
  renderAttentionMap(result.patch_attention, result.focus_region, result.prediction);
  renderShapChart(result.shap_feature_groups);
}

function renderAttentionMap(weights, focusRegion, prediction) {
  const card = el("attentionCard");
  if (!weights || !previewImg.src) {
    card.classList.add("hidden");
    return;
  }
  card.classList.remove("hidden");

  el("attentionImg").src = previewImg.src;

  const topIdx = weights.reduce((best, w, i) => (w > weights[best] ? i : best), 0);

  const focusText = el("focusRegionText");
  const reasonText = el("focusReasonText");
  if (focusRegion) {
    const pct = (focusRegion.confidence * 100).toFixed(0);
    focusText.innerHTML = `When evaluating this image as <b>${prediction}</b>, the model focused most on the <b>${focusRegion.jaw} · ${focusRegion.zone}</b> region (attention: ${pct}%).`;

    reasonText.textContent = focusRegion.reason || "";
    reasonText.classList.toggle("hidden", !focusRegion.reason);
  } else {
    focusText.innerHTML = "";
    reasonText.textContent = "";
  }

  const grid = el("attentionGrid");
  grid.innerHTML = "";
  weights.forEach((w, i) => {
    const cell = document.createElement("div");
    cell.className = "attention-cell" + (i === topIdx ? " is-top" : "");
    const opacity = 0.04 + Math.max(0, Math.min(1, w)) * 0.62;
    cell.style.setProperty("--cell-color", `rgba(235, 104, 52, ${opacity.toFixed(3)})`);
    grid.appendChild(cell);
  });
}

function renderShapChart(shapGroups) {
  // The detailed group/class/patch charts now live in the in-page screen
  // opened by the "SHAP Analysis" button (see renderShapReportBody) - here
  // we only decide whether that button should be visible.
  el("shapCard").classList.toggle("hidden", !shapGroups);
}

function renderProbChart(probabilities) {
  const chart = el("probChart");
  chart.innerHTML = "";

  CLASS_ORDER.forEach((className) => {
    const value = probabilities[className] ?? 0;
    const pct = (value * 100).toFixed(1);
    const color = CLASS_COLORS[className] || "#2a78d6";

    const row = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `
      <div class="prob-row-head">
        <span class="prob-row-name" style="--dot-color:${color}">${className}</span>
        <span class="prob-row-pct">${pct}%</span>
      </div>
      <div class="prob-track">
        <div class="prob-fill" style="--fill-color:${color}"></div>
      </div>
    `;
    chart.appendChild(row);
    // Apply the width animation on the next frame.
    requestAnimationFrame(() => {
      row.querySelector(".prob-fill").style.width = `${pct}%`;
    });
  });
}

function renderFindings(findings) {
  const list = el("findingsList");
  list.innerHTML = "";
  (findings || []).forEach((text) => {
    const li = document.createElement("li");
    li.innerHTML = `
      <svg class="finding-icon" width="16" height="16" viewBox="0 0 24 24" fill="none">
        <circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="1.7"/>
        <path d="M8.5 12.2l2.3 2.3 4.7-5" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
      <span>${text}</span>
    `;
    list.appendChild(li);
  });
}

// ---------------------------------------------------------------
// In-page screen transitions (SHAP report, Detailed Guidance) — inside the
// desktop app window (pywebview), window.open+blob URL doesn't work and
// shows a "get an app to open this link" error, so these are rendered as
// animated in-page "screens" instead of separate tabs/windows.
// ---------------------------------------------------------------

const mainLayout = el("mainLayout");
const SCREEN_TRANSITION_MS = 220;

let currentScreen = mainLayout;

function switchToScreen(targetEl) {
  const fromEl = currentScreen;
  if (fromEl === targetEl) return;

  fromEl.classList.add("screen-leave");
  setTimeout(() => {
    fromEl.classList.add("hidden");
    fromEl.classList.remove("screen-leave");

    targetEl.classList.remove("hidden");
    targetEl.classList.add("screen-enter-prep");
    void targetEl.offsetHeight; // force reflow so removing the class below animates
    targetEl.classList.remove("screen-enter-prep");
    window.scrollTo(0, 0);
  }, SCREEN_TRANSITION_MS);

  currentScreen = targetEl;
}

function returnToMain() {
  switchToScreen(mainLayout);
}

// --- SHAP Report ---

const shapReportScreen = el("shapReportScreen");
const shapReportBody = el("shapReportBody");
const shapReportBackBtn = el("shapReportBackBtn");
if (shapReportBackBtn) shapReportBackBtn.addEventListener("click", returnToMain);

const shapReportBtn = el("shapReportBtn");
if (shapReportBtn) {
  shapReportBtn.addEventListener("click", async () => {
    if (!selectedFile) return;
    const originalLabel = shapReportBtn.textContent;
    shapReportBtn.disabled = true;
    shapReportBtn.textContent = "Preparing SHAP report… (may take a few seconds)";
    try {
      const formData = new FormData();
      formData.append("image", selectedFile);
      const res = await fetch(API_SHAP_REPORT, { method: "POST", body: formData });
      if (!res.ok) throw new Error(await readErrorDetail(res));
      const data = await res.json();
      shapReportBody.innerHTML = renderShapReportBody(data, caseIdInput.value);
      switchToScreen(shapReportScreen);
    } catch (err) {
      showError(`Could not generate SHAP report: ${err.message}`);
    } finally {
      shapReportBtn.disabled = false;
      shapReportBtn.textContent = originalLabel;
    }
  });
}

// --- Detailed Clinical Guidance (no extra network request needed — already
// included in the /api/predict response as detailed_guidance) ---

const guidanceScreen = el("guidanceScreen");
const guidanceBody = el("guidanceBody");
const guidanceBackBtn = el("guidanceBackBtn");
const guidanceBtn = el("guidanceBtn");
if (guidanceBackBtn) guidanceBackBtn.addEventListener("click", returnToMain);

if (guidanceBtn) {
  guidanceBtn.addEventListener("click", () => {
    if (!lastResult || !lastResult.detailed_guidance) return;
    guidanceBody.innerHTML = renderGuidanceBody(lastResult.detailed_guidance, lastResult.prediction);
    switchToScreen(guidanceScreen);
  });
}

function renderGuidanceBody(guidance, prediction) {
  const nextStepsHtml = (guidance.next_steps || [])
    .map(
      (step, i) => `
    <li><span class="guidance-step-num">${i + 1}</span><span>${step}</span></li>`
    )
    .join("");

  const diffHtml = (guidance.differential_considerations || [])
    .map((item) => `<li>${item}</li>`)
    .join("");

  return `
  <div class="card">
    <h2>${prediction} — Overview</h2>
    <p class="guidance-overview">${guidance.overview}</p>
  </div>

  <div class="card">
    <h2>Recommended Next Steps</h2>
    <ul class="guidance-list">${nextStepsHtml}</ul>
  </div>

  <div class="card">
    <h2>Differential Considerations</h2>
    <ul class="guidance-diff-list">${diffHtml}</ul>
  </div>

  <div class="card">
    <h2>When to Escalate</h2>
    <div class="guidance-callout guidance-callout-escalation">${guidance.escalation_note}</div>
  </div>

  <div class="card">
    <h2>Note on Model Confidence</h2>
    <div class="guidance-callout guidance-callout-confidence">${guidance.confidence_note}</div>
  </div>

  <p class="disclaimer">Note: This guidance provides general, textbook-level information; a patient-specific
  definitive diagnosis and treatment decision requires clinician evaluation. This is not an approved medical
  device output.</p>`;
}

function renderShapReportBody(data, caseId) {
  const ts = new Date().toLocaleString("en-US");
  const predGroups = data.shap_per_class[data.prediction] || {};
  const groupEntries = SHAP_GROUP_ORDER.map((name) => [name, predGroups[name] || 0]);

  const barRows = groupEntries
    .map(
      ([name, pct]) => `
    <div class="prob-row">
      <div class="prob-row-head">
        <span class="prob-row-name" style="--dot-color:${SHAP_GROUP_COLORS[name]}">${name}</span>
        <span class="prob-row-pct">${pct.toFixed(1)}%</span>
      </div>
      <div class="prob-track"><div class="prob-fill" style="width:${pct}%; --fill-color:${SHAP_GROUP_COLORS[name]}"></div></div>
    </div>`
    )
    .join("");

  let cum = 0;
  const gradientStops = groupEntries
    .map(([name, pct]) => {
      const start = cum;
      cum += pct;
      return `${SHAP_GROUP_COLORS[name]} ${start}% ${cum}%`;
    })
    .join(", ");
  const legendHtml = groupEntries
    .map(
      ([name, pct]) => `
    <div class="shap-legend-item"><span class="dot" style="background:${SHAP_GROUP_COLORS[name]}"></span>${name} — <b>${pct.toFixed(1)}%</b></div>`
    )
    .join("");

  const maxVal = Math.max(
    ...CLASS_ORDER.flatMap((c) => SHAP_GROUP_ORDER.map((g) => (data.shap_per_class[c] || {})[g] || 0))
  );
  const classHeaderRow = CLASS_ORDER.map((c) => `<th>${c}${c === data.prediction ? " ★" : ""}</th>`).join("");
  const heatmapRows = SHAP_GROUP_ORDER.map((g) => {
    const cells = CLASS_ORDER.map((c) => {
      const v = (data.shap_per_class[c] || {})[g] || 0;
      const alpha = (v / (maxVal || 1)) * 0.75 + 0.06;
      const isPred = c === data.prediction;
      return `<td style="background:rgba(42,120,214,${alpha.toFixed(2)})${isPred ? "; box-shadow: inset 0 0 0 2px #184f95;" : ""}">${v.toFixed(1)}%</td>`;
    }).join("");
    return `<tr><th>${g}</th>${cells}</tr>`;
  }).join("");

  const patchPairs = (data.patch_attention || [])
    .map((w, i) => ({ w, label: (data.patch_labels || [])[i] || `Patch ${i + 1}` }))
    .sort((a, b) => b.w - a.w);
  const patchRows = patchPairs
    .map(
      (p, rank) => `
    <div class="prob-row">
      <div class="prob-row-head">
        <span class="prob-row-name">${rank === 0 ? "🔴 " : ""}${p.label}</span>
        <span class="prob-row-pct">${(p.w * 100).toFixed(0)}%</span>
      </div>
      <div class="prob-track"><div class="prob-fill" style="width:${p.w * 100}%; --fill-color:#eb6834"></div></div>
    </div>`
    )
    .join("");

  const probsHtml = CLASS_ORDER.map(
    (c) => `<li><b>${c}:</b> ${((data.probabilities[c] || 0) * 100).toFixed(2)}%</li>`
  ).join("");

  return `
  <div>
    <p class="disclaimer" style="text-align:left; padding:0;">Case: <b>${caseId || "-"}</b> &nbsp;·&nbsp; Generated: ${ts} &nbsp;·&nbsp; Model prediction: <b>${data.prediction}</b></p>
  </div>

  <div class="card">
    <h2>Class Probabilities</h2>
    <ul class="shap-probs-list">${probsHtml}</ul>
  </div>

  <div class="card">
    <h2>Feature Group Contribution — ${data.prediction}</h2>
    <p class="hint">Shows which feature groups the SVM's prediction for this case relied on (Shapley values, computed on the PCA85 model and projected back onto the groups).</p>
    ${barRows}
  </div>

  <div class="card">
    <h2>Same Distribution — Pie Chart</h2>
    <div class="shap-pie-row">
      <div class="shap-pie" style="background: conic-gradient(${gradientStops})"></div>
      <div class="shap-legend">${legendHtml}</div>
    </div>
  </div>

  <div class="card">
    <h2>Class-by-Class Comparison</h2>
    <p class="hint">This case's feature vector has been explained with SHAP separately for each of the 4 classes (★ = the model's actual prediction). Darker cells indicate that feature group was relatively more influential for that class.</p>
    <table class="shap-table">
      <thead><tr><th></th>${classHeaderRow}</tr></thead>
      <tbody>${heatmapRows}</tbody>
    </table>
  </div>

  <div class="card">
    <h2>Patch Importance Ranking</h2>
    <p class="hint">The 16 patches (2×8 grid), ranked from highest to lowest by the model's attention weight. The top one (🔴) is the patch marked with a red border on the attention map.</p>
    ${patchRows}
  </div>

  <p class="disclaimer">Note: This report is computed live for a single case; the academic paper's SHAP analyses (13_1002.py) show the aggregate distribution over the full test set, whereas the percentages here are approximate, case-specific contributions. Clinician evaluation is required for a definitive diagnosis.</p>`;
}

function downloadBlob(content, filename, mime) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

el("downloadJsonBtn").addEventListener("click", () => {
  if (!lastResult) return;
  downloadBlob(
    JSON.stringify(lastResult, null, 2),
    `${lastResult.case.case_id || "DSS"}_Output.json`,
    "application/json"
  );
});

el("downloadHtmlBtn").addEventListener("click", () => {
  if (!lastResult) return;
  downloadBlob(
    buildReportHtml(lastResult),
    `${lastResult.case.case_id || "DSS"}_Report.html`,
    "text/html"
  );
});

const TRIAGE_STATUS_COLORS = {
  good: "#0ca30c",
  warning: "#d98c00",
  serious: "#d9622f",
  critical: "#d03b3b",
};

function buildReportHtml(result) {
  const ts = new Date(result.timestamp).toLocaleString("en-US");
  const c = result.case || {};
  const predColor = CLASS_COLORS[result.prediction] || "#2a78d6";
  const triageColor = TRIAGE_STATUS_COLORS[result.triage_level] || "#8a97a6";
  const confidencePct = (result.confidence * 100).toFixed(1);

  const barRows = CLASS_ORDER.map((className) => {
    const value = result.probabilities[className] || 0;
    const pct = (value * 100).toFixed(1);
    const color = CLASS_COLORS[className] || "#2a78d6";
    const isPred = className === result.prediction;
    return `
    <div class="bar-row">
      <div class="bar-row-head">
        <span class="bar-row-name"><span class="dot" style="background:${color}"></span>${className}${isPred ? " ★" : ""}</span>
        <span class="bar-row-pct">${pct}%</span>
      </div>
      <div class="bar-track"><div class="bar-fill" style="width:${pct}%; background:${color}"></div></div>
    </div>`;
  }).join("");

  const findingsHtml = (result.findings || [])
    .map(
      (f) => `
    <li>
      <svg width="17" height="17" viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <circle cx="12" cy="12" r="9" stroke="#2a78d6" stroke-width="1.7"/>
        <path d="M8.5 12.2l2.3 2.3 4.7-5" stroke="#2a78d6" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
      <span>${f}</span>
    </li>`
    )
    .join("");

  const shapEntries = result.shap_feature_groups
    ? SHAP_GROUP_ORDER.map((name) => [name, result.shap_feature_groups[name] || 0])
    : null;
  const shapHtml = shapEntries
    ? `
  <div class="card">
    <h2>SHAP Analysis — Feature Group Contribution</h2>
    <p class="hint">Shows which feature groups the SVM's prediction for this case relied on. For a detailed report including a class-by-class comparison and patch importance ranking, use the "SHAP Analysis" button in the app.</p>
    ${shapEntries
      .map(
        ([name, pct]) => `
    <div class="bar-row">
      <div class="bar-row-head">
        <span class="bar-row-name"><span class="dot" style="background:${SHAP_GROUP_COLORS[name]}"></span>${name}</span>
        <span class="bar-row-pct">${pct.toFixed(1)}%</span>
      </div>
      <div class="bar-track"><div class="bar-fill" style="width:${pct}%; background:${SHAP_GROUP_COLORS[name]}"></div></div>
    </div>`
      )
      .join("")}
  </div>`
    : "";

  return `<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>DSS Report — ${c.case_id || ""}</title>
<style>
  :root {
    --primary: #2a78d6; --primary-dark: #184f95; --primary-light: #e9f1fc;
    --text-primary: #10233d; --text-secondary: #4c6178; --text-muted: #8a97a6;
    --border: rgba(16,58,99,0.10); --surface: #fff; --surface-alt: #f4f8fc;
  }
  * { box-sizing: border-box; }
  body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--text-primary);
    background: linear-gradient(180deg, #e9f1fb 0%, #f7fafd 420px); margin: 0; padding: 0 20px 60px; }
  .wrap { max-width: 860px; margin: 0 auto; display: flex; flex-direction: column; gap: 18px; }

  .hero-banner {
    margin: 0 -20px 4px; padding: 28px 32px;
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
    color: #fff; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 14px;
    box-shadow: 0 10px 28px rgba(24,79,149,0.25);
  }
  .hero-brand { display: flex; align-items: center; gap: 14px; }
  .hero-title { font-size: 1.25rem; font-weight: 800; letter-spacing: -0.01em; }
  .hero-title span { background: rgba(255,255,255,0.22); padding: 2px 9px; border-radius: 999px; font-size: 0.7rem; margin-left: 6px; vertical-align: middle; }
  .hero-sub { font-size: 0.82rem; opacity: 0.88; margin-top: 2px; }
  .hero-meta { text-align: right; font-size: 0.82rem; opacity: 0.92; line-height: 1.6; }
  .hero-meta b { font-weight: 700; }

  .result-hero {
    display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 18px;
    background: var(--surface); border: 1px solid var(--border); border-radius: 16px;
    box-shadow: 0 1px 2px rgba(16,58,99,0.06); padding: 24px 28px;
  }
  .result-label { display: block; font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); margin-bottom: 6px; }
  .result-value { font-size: 1.9rem; font-weight: 800; letter-spacing: -0.01em; }
  .triage-pill {
    display: inline-flex; align-items: center; gap: 7px; margin-top: 10px; padding: 5px 13px;
    border-radius: 999px; font-size: 0.78rem; font-weight: 700; color: #fff;
  }
  .triage-pill .pill-dot { width: 7px; height: 7px; border-radius: 50%; background: #fff; }
  .conf-block { text-align: center; padding-left: 24px; border-left: 1px solid var(--border); }
  .conf-num { display: block; font-size: 2.1rem; font-weight: 800; color: var(--primary-dark); line-height: 1; }
  .conf-label { font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); }

  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
  @media (max-width: 640px) { .grid-2 { grid-template-columns: 1fr; } .hero-meta { text-align: left; } }

  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
    box-shadow: 0 1px 2px rgba(16,58,99,0.06); padding: 22px 24px; }
  .card h2 { font-size: 1rem; margin: 0 0 14px; color: var(--primary-dark); }
  .hint { font-size: 0.8rem; color: var(--text-secondary); margin: -6px 0 14px; line-height: 1.5; }

  .case-grid { display: grid; grid-template-columns: auto 1fr; gap: 9px 14px; font-size: 0.86rem; margin: 0; }
  .case-grid dt { color: var(--text-muted); font-weight: 600; }
  .case-grid dd { margin: 0; font-weight: 600; }

  .bar-row { display: flex; flex-direction: column; gap: 6px; margin-bottom: 14px; }
  .bar-row:last-child { margin-bottom: 0; }
  .bar-row-head { display: flex; justify-content: space-between; font-size: 0.85rem; }
  .bar-row-name { font-weight: 700; display: flex; align-items: center; gap: 7px; }
  .bar-row-pct { font-weight: 800; color: var(--text-secondary); }
  .bar-track { height: 11px; border-radius: 999px; background: var(--surface-alt); border: 1px solid var(--border); overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 999px; }
  .dot { width: 9px; height: 9px; border-radius: 3px; display: inline-block; flex-shrink: 0; }

  .findings { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 11px; }
  .findings li { display: flex; gap: 10px; align-items: flex-start; font-size: 0.9rem; color: var(--text-secondary); line-height: 1.55; }
  .findings svg { flex-shrink: 0; margin-top: 2px; }

  .suggestion-card { background: linear-gradient(135deg, var(--primary-light) 0%, #ffffff 100%); border-left: 4px solid var(--primary); }
  .suggestion-card p { font-size: 0.94rem; line-height: 1.65; margin: 0; }

  .print-btn {
    align-self: center; margin-top: 4px; padding: 11px 22px; border-radius: 10px; border: 1px solid rgba(42,120,214,0.25);
    background: var(--primary-light); color: var(--primary-dark); font-weight: 700; font-size: 0.86rem; cursor: pointer;
  }
  .print-btn:hover { background: #dcebfb; }

  .disclaimer { font-size: 0.76rem; color: var(--text-muted); text-align: center; line-height: 1.6; padding: 4px 10px 20px; }

  @media print {
    body { background: #fff; }
    .hero-banner { box-shadow: none; }
    .card, .result-hero { box-shadow: none; }
    .print-btn { display: none; }
  }
</style></head>
<body><div class="wrap">

  <div class="hero-banner">
    <div class="hero-brand">
      <svg width="42" height="42" viewBox="0 0 48 48" fill="none" aria-hidden="true">
        <rect x="1" y="1" width="46" height="46" rx="13" fill="rgba(255,255,255,0.16)"/>
        <path d="M24 13c-3.2 0-5.1 1.7-6.9 1.7-1.6 0-2.8-1-4.3-1-2.6 0-5.3 2.2-5.3 6.9 0 4.3 3.1 12.4 5.6 12.4 1.5 0 2-1 3.3-1 1.3 0 1.6 1 3.3 1 1.1 0 1.6-1.1 2.1-2.4.2-.6 1.1-.6 1.4 0 .5 1.3 1 2.4 2.1 2.4 1.7 0 2-1 3.3-1 1.3 0 1.8 1 3.3 1 2.5 0 5.6-8.1 5.6-12.4 0-4.7-2.7-6.9-5.3-6.9-1.5 0-2.7 1-4.3 1-1.8 0-3.7-1.7-6.9-1.7Z" fill="#fff"/>
        <circle cx="24" cy="23" r="2.1" fill="#2a78d6"/>
      </svg>
      <div>
        <div class="hero-title">PatchDentalNet <span>DSS</span></div>
        <div class="hero-sub">Panoramic Radiograph Decision Support System Report</div>
      </div>
    </div>
    <div class="hero-meta">
      <div>Case: <b>${c.case_id || "-"}</b></div>
      <div>${ts}</div>
    </div>
  </div>

  <div class="result-hero">
    <div>
      <span class="result-label">Model Prediction</span>
      <span class="result-value" style="color:${predColor}">${result.prediction}</span>
      <div><span class="triage-pill" style="background:${triageColor}"><span class="pill-dot"></span>Triage: ${result.triage}</span></div>
    </div>
    <div class="conf-block">
      <span class="conf-num">${confidencePct}%</span>
      <span class="conf-label">Confidence</span>
    </div>
  </div>

  <div class="grid-2">
    <div class="card">
      <h2>Case Information</h2>
      <dl class="case-grid">
        <dt>Case ID</dt><dd>${c.case_id || "-"}</dd>
        <dt>Age</dt><dd>${c.age || "-"}</dd>
        <dt>Gender</dt><dd>${c.gender || "-"}</dd>
        <dt>Complaint</dt><dd>${c.complaint || "-"}</dd>
        <dt>Notes</dt><dd>${c.notes || "-"}</dd>
      </dl>
    </div>
    <div class="card">
      <h2>Class Probabilities</h2>
      ${barRows}
    </div>
  </div>

  <div class="card">
    <h2>DSS Findings</h2>
    <ul class="findings">${findingsHtml}</ul>
  </div>

  <div class="card suggestion-card">
    <h2>Suggestion</h2>
    <p>${result.suggestion}</p>
  </div>

  ${shapHtml}

  <button class="print-btn" onclick="window.print()">🖨️ Print / Save as PDF</button>

  <p class="disclaimer">Note: This output is a decision support suggestion; clinician evaluation is required for a definitive diagnosis. (Model mode: ${result.model_mode})</p>

</div></body></html>`;
}
