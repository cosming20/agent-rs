/**
 * Agentic RAG — Document ingestion module.
 * Handles drag-and-drop file upload, progress tracking, and result display.
 */

let api, utils;
let selectedFile = null;

const ALLOWED_EXT = new Set(['.pdf', '.docx', '.pptx', '.html', '.htm', '.txt', '.md']);
const MAX_SIZE = 50 * 1024 * 1024;

export function initIngest(apiClient, shared) {
  api = apiClient;
  utils = shared;

  const zone = document.getElementById('upload-zone');
  const fileInput = document.getElementById('file-input');
  const uploadBtn = document.getElementById('upload-btn');
  const removeBtn = document.getElementById('file-remove');

  // Click to browse
  zone.addEventListener('click', () => fileInput.click());

  // File input change
  fileInput.addEventListener('change', () => {
    if (fileInput.files.length) selectFile(fileInput.files[0]);
  });

  // Drag and drop
  zone.addEventListener('dragover', e => {
    e.preventDefault();
    zone.classList.add('dragover');
  });

  zone.addEventListener('dragleave', e => {
    e.preventDefault();
    zone.classList.remove('dragover');
  });

  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('dragover');
    if (e.dataTransfer.files.length) selectFile(e.dataTransfer.files[0]);
  });

  // Upload
  uploadBtn.addEventListener('click', startUpload);

  // Remove file
  removeBtn.addEventListener('click', resetForm);
}


/* ═══════ FILE SELECTION ═══════ */

function selectFile(file) {
  const ext = '.' + file.name.split('.').pop().toLowerCase();

  if (!ALLOWED_EXT.has(ext)) {
    showResult({
      success: false,
      message: `Unsupported file type "${ext}". Allowed: ${[...ALLOWED_EXT].join(', ')}`,
    });
    return;
  }

  if (file.size > MAX_SIZE) {
    showResult({
      success: false,
      message: `File exceeds ${MAX_SIZE / (1024 * 1024)} MB limit`,
    });
    return;
  }

  selectedFile = file;

  document.getElementById('file-name').textContent = file.name;
  document.getElementById('file-size').textContent = utils.formatFileSize(file.size);
  document.getElementById('file-info').classList.remove('hidden');
  document.getElementById('upload-btn').disabled = false;
  document.getElementById('upload-zone').classList.add('has-file');

  // Hide any previous result
  document.getElementById('result-card').classList.add('hidden');
  document.getElementById('progress-track').classList.add('hidden');
}


/* ═══════ UPLOAD ═══════ */

async function startUpload() {
  if (!selectedFile) return;

  const uploadBtn = document.getElementById('upload-btn');
  uploadBtn.disabled = true;
  uploadBtn.classList.add('loading');
  uploadBtn.innerHTML = `
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="animation: spin 0.7s linear infinite">
      <path d="M21 12a9 9 0 1 1-6.219-8.56"/>
    </svg>
    Processing...`;

  // Show progress
  const track = document.getElementById('progress-track');
  track.classList.remove('hidden');
  setStep('upload', 'active');

  try {
    // Simulate stepping through stages
    await delay(300);
    setStep('upload', 'done');
    setStep('parse', 'active');

    const result = await api.upload(selectedFile);

    setStep('parse', 'done');
    setStep('cognify', 'active');
    await delay(200);
    setStep('cognify', 'done');
    setStep('done', 'done');

    showResult({
      success: true,
      data: result,
    });
  } catch (err) {
    // Mark current active step as error
    document.querySelectorAll('.progress-step.active').forEach(s => {
      s.classList.remove('active');
      s.classList.add('error');
    });

    showResult({
      success: false,
      message: err.message,
    });
  } finally {
    uploadBtn.classList.remove('loading');
    resetUploadBtn();
  }
}


/* ═══════ PROGRESS ═══════ */

function setStep(name, state) {
  const step = document.querySelector(`.progress-step[data-step="${name}"]`);
  if (!step) return;
  step.classList.remove('active', 'done', 'error');
  step.classList.add(state);
}

function resetSteps() {
  document.querySelectorAll('.progress-step').forEach(s => {
    s.classList.remove('active', 'done', 'error');
  });
}


/* ═══════ RESULT ═══════ */

function showResult({ success, data, message }) {
  const card = document.getElementById('result-card');
  card.classList.remove('hidden', 'success', 'error');
  card.classList.add(success ? 'success' : 'error');

  if (success && data) {
    const docId = data.parsed_document_id || '—';
    card.innerHTML = `
      <div class="result-header">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><path d="m9 11 3 3L22 4"/>
        </svg>
        <span class="result-title">Ingestion complete</span>
      </div>
      <div class="result-stats">
        <div class="result-stat">
          <div class="result-stat-label">Chunks</div>
          <div class="result-stat-value">${data.chunk_count}</div>
        </div>
        <div class="result-stat">
          <div class="result-stat-label">Cognee</div>
          <div class="result-stat-value">${data.cognee_ingested ? 'Indexed' : 'Skipped'}</div>
        </div>
        <div class="result-stat">
          <div class="result-stat-label">Document ID</div>
          <div class="result-stat-value" style="font-size:11px;word-break:break-all">${docId}</div>
        </div>
        <div class="result-stat">
          <div class="result-stat-label">File</div>
          <div class="result-stat-value" style="font-size:12px">${data.filename || '—'}</div>
        </div>
      </div>
      <div class="result-actions">
        <button class="btn-secondary" id="upload-another">Upload another</button>
      </div>`;
    document.getElementById('upload-another')?.addEventListener('click', resetForm);
  } else {
    card.innerHTML = `
      <div class="result-header">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"/><path d="m15 9-6 6"/><path d="m9 9 6 6"/>
        </svg>
        <span class="result-title">Ingestion failed</span>
      </div>
      <p style="font-size:13px;color:var(--text-secondary);margin-top:4px">${message || 'Unknown error'}</p>
      <div class="result-actions">
        <button class="btn-secondary" id="try-again">Try again</button>
      </div>`;
    document.getElementById('try-again')?.addEventListener('click', resetForm);
  }
}


/* ═══════ RESET ═══════ */

function resetForm() {
  selectedFile = null;
  document.getElementById('file-input').value = '';
  document.getElementById('file-info').classList.add('hidden');
  document.getElementById('result-card').classList.add('hidden');
  document.getElementById('progress-track').classList.add('hidden');
  document.getElementById('upload-zone').classList.remove('has-file');
  resetUploadBtn();
  resetSteps();
}

function resetUploadBtn() {
  const btn = document.getElementById('upload-btn');
  btn.disabled = !selectedFile;
  btn.innerHTML = `
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
      <polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/>
    </svg>
    Upload &amp; Ingest`;
}


/* ═══════ UTILS ═══════ */

function delay(ms) {
  return new Promise(r => setTimeout(r, ms));
}
