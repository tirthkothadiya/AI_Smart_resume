/**
 * ResumeIQ — Frontend Logic
 * Handles: drag-and-drop, file list, weight sync, API call, results rendering
 */

// ─── State ────────────────────────────────────────────────────────────────────

let selectedFiles = []; // Tracks currently selected PDF files

// ─── Drag & Drop ─────────────────────────────────────────────────────────────

const dropZone = document.getElementById("dropZone");
const resumeInput = document.getElementById("resumeInput");

dropZone.addEventListener("click", () => resumeInput.click());

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragging");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragging"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragging");
  addFiles([...e.dataTransfer.files]);
});

resumeInput.addEventListener("change", () => {
  addFiles([...resumeInput.files]);
  resumeInput.value = ""; // Reset so same file can be re-added after removal
});

/**
 * Add files to selectedFiles[], skip duplicates and non-PDFs.
 */
function addFiles(files) {
  files.forEach((file) => {
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      showInlineError(`"${file.name}" is not a PDF — skipped.`);
      return;
    }
    if (selectedFiles.some((f) => f.name === file.name && f.size === file.size)) return;
    selectedFiles.push(file);
  });
  renderFileList();
}

function removeFile(index) {
  selectedFiles.splice(index, 1);
  renderFileList();
}

function renderFileList() {
  const list = document.getElementById("fileList");
  list.innerHTML = "";
  selectedFiles.forEach((file, i) => {
    const li = document.createElement("li");
    li.className = "file-item";
    li.innerHTML = `
      <span class="fi-icon">📄</span>
      <span class="fi-name" title="${esc(file.name)}">${esc(file.name)}</span>
      <span class="fi-size">${formatBytes(file.size)}</span>
      <button class="fi-remove" title="Remove" onclick="removeFile(${i})">✕</button>
    `;
    list.appendChild(li);
  });
}

// ─── Weight Sliders ───────────────────────────────────────────────────────────

/**
 * Keep tfidf + sbert weights summing to 100.
 */
function syncWeights(changedSlider, otherKey) {
  const val = parseInt(changedSlider.value, 10);
  const other = document.getElementById(`${otherKey}Weight`);
  other.value = 100 - val;
  document.getElementById("tfidfVal").textContent    = `${document.getElementById("tfidfWeight").value}%`;
  document.getElementById("sbertVal").textContent = `${document.getElementById("sbertWeight").value}%`;
}

// ─── Inline error (transient) ─────────────────────────────────────────────────

function showInlineError(msg) {
  const box = document.getElementById("errorBox");
  box.textContent = "⚠ " + msg;
  box.hidden = false;
  document.getElementById("placeholder").hidden = true;
  document.getElementById("tableWrap").hidden   = true;
  document.getElementById("legend").hidden      = true;
}

// ─── Analysis ─────────────────────────────────────────────────────────────────

async function runAnalysis() {
  const jobDesc = document.getElementById("jobDesc").value.trim();
  const btn     = document.getElementById("analyzeBtn");

  // Basic client-side validation
  if (selectedFiles.length === 0) { showInlineError("Please upload at least one PDF resume."); return; }
  if (!jobDesc)                   { showInlineError("Please enter a job description."); return; }

  // UI: show loader
  btn.disabled = true;
  setResultsState("loading");

  // Build FormData
  const formData = new FormData();
  selectedFiles.forEach((f) => formData.append("resumes", f));
  formData.append("job_description",  jobDesc);
  formData.append("tfidf_weight",     (parseInt(document.getElementById("tfidfWeight").value, 10) / 100).toFixed(2));
  formData.append("sbert_weight",  (parseInt(document.getElementById("sbertWeight").value, 10) / 100).toFixed(2));

  try {
    const res  = await fetch("/analyze", { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok || data.error) {
      showInlineError(data.error || "An unexpected error occurred.");
      setResultsState("error");
    } else {
      renderResults(data.results);
    }
  } catch (err) {
    showInlineError("Network error — is the Flask server running?");
    setResultsState("error");
  } finally {
    btn.disabled = false;
  }
}

/** Toggle visibility of results panel sections. */
function setResultsState(state) {
  document.getElementById("placeholder").hidden  = state !== "empty";
  document.getElementById("loaderWrap").hidden   = state !== "loading";
  document.getElementById("errorBox").hidden     = state !== "error";
  document.getElementById("tableWrap").hidden    = state !== "results";
  document.getElementById("legend").hidden       = state !== "results";
  document.getElementById("resultsBadge").hidden = state !== "results";
}

// ─── Render Results ───────────────────────────────────────────────────────────

function renderResults(results) {
  setResultsState("results");

  const badge = document.getElementById("resultsBadge");
  badge.textContent = `${results.length} candidate${results.length !== 1 ? "s" : ""}`;
  badge.hidden = false;

  const tbody = document.getElementById("tableBody");
  tbody.innerHTML = "";

  results.forEach((r) => {
    const isTop = r.rank <= 3;
    const tr = document.createElement("tr");
    if (isTop) tr.classList.add("top-candidate");

    tr.innerHTML = `
      <td class="rank-cell">
        <span class="rank-badge rank-${r.rank <= 3 ? r.rank : "other"}">${r.rank}</span>
      </td>
      <td>
        <span class="fname" title="${esc(r.filename)}">${esc(r.filename)}</span>
      </td>
      <td>
        ${scoreBar(r.tfidf_score)}
      </td>
      <td>
        ${scoreBar(r.sbert_score)}
      </td>
      <td>
        <div class="combined-score">${r.combined_score}<span class="combined-pct">%</span></div>
      </td>
      <td>
        ${renderKeywords(r.matched_keywords)}
      </td>
      <td>
        <button class="dl-btn" onclick="downloadResume('${esc(r.filename)}')">↓ Download</button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function scoreBar(score) {
  const pct = Math.min(score, 100);
  return `
    <div class="score-wrap">
      <div class="score-bar-bg"><div class="score-bar-fill" style="width:${pct}%"></div></div>
      <span class="score-num">${score}%</span>
    </div>
  `;
}

function renderKeywords(kws) {
  if (!kws || kws.length === 0) return `<span class="kw-more">—</span>`;
  const shown   = kws.slice(0, 5);
  const hidden  = kws.length - shown.length;
  let html = `<div class="kw-list">` + shown.map(k => `<span class="kw-tag">${esc(k)}</span>`).join("") + `</div>`;
  if (hidden > 0) html += `<span class="kw-more">+${hidden} more</span>`;
  return html;
}

function downloadResume(filename) {
  window.open(`/download/${encodeURIComponent(filename)}`, "_blank");
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function esc(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}
