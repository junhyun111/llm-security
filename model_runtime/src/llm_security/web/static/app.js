const folderInput = document.querySelector('#folder-input');
const selection = document.querySelector('#selection');
const selectedName = document.querySelector('#selected-name');
const selectedMeta = document.querySelector('#selected-meta');
const startButton = document.querySelector('#start-button');
const uploadError = document.querySelector('#upload-error');
const statusPanel = document.querySelector('#status-panel');
const statusTitle = document.querySelector('#status-title');
const statusPill = document.querySelector('#status-pill');
const statusMessage = document.querySelector('#status-message');
const progressBar = document.querySelector('#progress-bar');
const progressValue = document.querySelector('#progress-value');
const jobError = document.querySelector('#job-error');
const resultsSection = document.querySelector('#results');
const metrics = document.querySelector('#metrics');
const findingsContainer = document.querySelector('#findings');
const findingTemplate = document.querySelector('#finding-template');
const downloadLink = document.querySelector('#download-link');
const selectionCount = document.querySelector('#selection-count');
const batchProposalButton = document.querySelector('#batch-proposal-button');
const batchPatchPanel = document.querySelector('#batch-patch-panel');
const batchPatchTitle = document.querySelector('#batch-patch-title');
const batchPatchStatus = document.querySelector('#batch-patch-status');
const batchPatchSummary = document.querySelector('#batch-patch-summary');
const batchPatchDiff = document.querySelector('#batch-patch-diff');
const batchActions = document.querySelector('#batch-actions');

let selectedFiles = [];
let currentJob = null;
let analysisResult = null;
let activeFilter = 'all';
const selectedFindingIds = new Set();

folderInput.addEventListener('change', () => {
  selectedFiles = Array.from(folderInput.files || []);
  uploadError.hidden = true;
  if (!selectedFiles.length) {
    selection.hidden = true;
    return;
  }
  const root = selectedFiles[0].webkitRelativePath.split('/')[0] || 'project';
  const total = selectedFiles.reduce((sum, file) => sum + file.size, 0);
  selectedName.textContent = root;
  selectedMeta.textContent = `${selectedFiles.length.toLocaleString()}개 파일 · ${formatBytes(total)}`;
  selection.hidden = false;
});

startButton.addEventListener('click', async () => {
  if (!selectedFiles.length) return;
  startButton.disabled = true;
  uploadError.hidden = true;
  const form = new FormData();
  const projectName = selectedFiles[0].webkitRelativePath.split('/')[0] || 'project';
  form.append('project_name', projectName);
  for (const file of selectedFiles) {
    form.append('relative_paths', file.webkitRelativePath || file.name);
    form.append('files', file, file.name);
  }
  try {
    const response = await fetch('/api/jobs', { method: 'POST', body: form });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || '업로드하지 못했습니다.');
    currentJob = payload;
    selectedFindingIds.clear();
    statusPanel.hidden = false;
    resultsSection.hidden = true;
    updateStatus(payload);
    statusPanel.scrollIntoView({ behavior: 'smooth', block: 'center' });
    pollJob();
  } catch (error) {
    uploadError.textContent = error.message;
    uploadError.hidden = false;
  } finally {
    startButton.disabled = false;
  }
});

async function pollJob() {
  if (!currentJob) return;
  try {
    const response = await fetch(`/api/jobs/${currentJob.job_id}`);
    const job = await response.json();
    if (!response.ok) throw new Error(job.detail || '작업 상태를 불러오지 못했습니다.');
    currentJob = job;
    updateStatus(job);
    if (job.status === 'completed') {
      await loadAnalysis();
      return;
    }
    if (job.status === 'failed') {
      jobError.textContent = job.error || '분석에 실패했습니다.';
      jobError.hidden = false;
      return;
    }
    setTimeout(pollJob, 1800);
  } catch (error) {
    jobError.textContent = error.message;
    jobError.hidden = false;
  }
}

function updateStatus(job) {
  const labels = {
    uploading: '업로드 중', queued: '분석 대기 중', analyzing: '분석 진행 중',
    completed: '분석 완료', failed: '분석 실패'
  };
  statusTitle.textContent = labels[job.status] || job.status;
  statusPill.textContent = job.status.toUpperCase();
  statusMessage.textContent = job.message || '';
  progressBar.style.width = `${job.progress || 0}%`;
  progressValue.textContent = `${job.progress || 0}%`;
}

async function loadAnalysis() {
  const response = await fetch(`/api/jobs/${currentJob.job_id}/analysis`);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || '결과를 불러오지 못했습니다.');
  analysisResult = payload;
  if (payload.patch_batch) {
    for (const findingId of payload.patch_batch.finding_ids) selectedFindingIds.add(findingId);
  }
  renderMetrics(payload.summary);
  renderFindings();
  renderBatchPatch();
  downloadLink.href = `/api/jobs/${currentJob.job_id}/download`;
  resultsSection.hidden = false;
  resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function renderMetrics(summary) {
  const values = [
    ['분석 소스', `${summary.source_file_count}개`],
    ['정적 후보', `${summary.candidate_count}개`],
    ['정적 CWE 후보', `${summary.cwe_hypothesis_count || 0}개`],
    ['취약점', `${summary.finding_count}개`],
    ['API 호출', `${summary.request_count || 0} / 1회`],
    ['전문가 작업', `${summary.submitted_expert_task_count || 0}개`],
    ['API 비용', `$${Number(summary.total_cost || 0).toFixed(4)}`],
  ];
  metrics.innerHTML = values.map(([label, value]) =>
    `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`
  ).join('');
}

function renderFindings() {
  findingsContainer.innerHTML = '';
  const rows = (analysisResult?.findings || []).filter((bundle) =>
    activeFilter === 'all' || bundle.validation.verdict === activeFilter
  );
  if (!rows.length) {
    findingsContainer.innerHTML = '<div class="empty">이 조건에 해당하는 탐지 내역이 없습니다.</div>';
    updateSelectionControls();
    return;
  }
  for (const bundle of rows) findingsContainer.append(renderFinding(bundle));
  updateSelectionControls();
}

function renderFinding(bundle) {
  const node = findingTemplate.content.cloneNode(true);
  const finding = bundle.finding;
  const validation = bundle.validation;
  const cweHypotheses = bundle.candidate?.cwe_hypotheses || [];
  node.querySelector('.finding-title').textContent = finding.title;
  node.querySelector('.location').textContent = `${finding.file}:${finding.line_start}-${finding.line_end} · ${finding.function}`;
  node.querySelector('.root-cause').textContent = finding.root_cause;
  node.querySelector('.consequence').textContent = finding.consequence;
  node.querySelector('.confidence').textContent = `${Math.round(validation.confidence * 100)}%`;
  const experts = finding.supporting_experts?.length ? finding.supporting_experts : [finding.expert];
  node.querySelector('.badges').innerHTML = [
    `<span class="badge ${validation.verdict}">${validation.verdict}</span>`,
    ...experts.map((expert) => `<span class="badge">${escapeHtml(expertLabel(expert))}</span>`),
    ...(finding.cwes || []).map((cwe) => `<span class="badge">${escapeHtml(cwe)}</span>`),
  ].join('');
  node.querySelector('.evidence-block').innerHTML = `
    <strong>로컬 검증 판단</strong><ul>${(validation.reasons || []).map((x) => `<li>${escapeHtml(x)}</li>`).join('')}</ul>
    <strong>정적 근거 ID</strong><p>${(finding.evidence_ids || []).map(escapeHtml).join(', ') || '없음'}</p>
    <strong>정적 CWE 후보 (LLM 검증 전)</strong><ul>${cweHypotheses.map((item) =>
      `<li>${escapeHtml(item.cwe)} · ${Math.round(Number(item.confidence || 0) * 100)}% · ${escapeHtml((item.reasons || []).join('; '))}</li>`
    ).join('') || '<li>없음</li>'}</ul>
    <strong>반증 근거</strong><ul>${(finding.evidence_against || []).map((x) => `<li>${escapeHtml(x)}</li>`).join('') || '<li>없음</li>'}</ul>
  `;
  const patchArea = node.querySelector('.patch-area');
  if (validation.verdict !== 'validated') {
    patchArea.innerHTML = '<span class="badge">패치 선택 대상 아님</span>';
  } else {
    const label = document.createElement('label');
    label.className = 'finding-selector';
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.checked = selectedFindingIds.has(finding.finding_id);
    checkbox.disabled = Boolean(analysisResult.patch_batch);
    checkbox.addEventListener('change', () => {
      if (checkbox.checked) selectedFindingIds.add(finding.finding_id);
      else selectedFindingIds.delete(finding.finding_id);
      updateSelectionControls();
    });
    label.append(checkbox, document.createTextNode(' 통합 패치에 포함'));
    patchArea.append(label);
  }
  return node;
}

function expertLabel(expert) {
  const labels = {
    memory_bounds: 'E1 Memory Safety',
    lifetime_resource: 'Legacy E2 Lifetime / Resource',
    integer_size_type: 'E3 Integer / Size / Type',
    taint_api_contract: 'E4 Taint / API Contract',
    control_state_error: 'E5 Control / State / Error',
    concurrency_toctou: 'E6 Concurrency / TOCTOU',
  };
  return labels[expert] || expert;
}

function updateSelectionControls() {
  selectionCount.textContent = `선택된 취약점 ${selectedFindingIds.size}개`;
  batchProposalButton.disabled = selectedFindingIds.size === 0 || Boolean(analysisResult?.patch_batch);
}

batchProposalButton.addEventListener('click', async () => {
  if (!selectedFindingIds.size || !currentJob) return;
  batchProposalButton.disabled = true;
  const old = batchProposalButton.textContent;
  batchProposalButton.textContent = '통합 수정안 생성 중…';
  try {
    const response = await fetch(`/api/jobs/${currentJob.job_id}/patches/proposal`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ finding_ids: Array.from(selectedFindingIds) }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || '통합 수정안을 생성하지 못했습니다.');
    analysisResult.patch_batch = payload;
    renderFindings();
    renderBatchPatch();
  } catch (error) {
    alert(error.message);
    batchProposalButton.disabled = false;
  } finally {
    batchProposalButton.textContent = old;
  }
});

function renderBatchPatch() {
  const patch = analysisResult?.patch_batch;
  batchPatchPanel.hidden = !patch;
  if (!patch) return;
  batchPatchTitle.textContent = `${patch.finding_ids.length}개 취약점 통합 수정안`;
  batchPatchStatus.textContent = patch.status.toUpperCase();
  batchPatchSummary.textContent = patch.summary;
  batchPatchDiff.textContent = patch.unified_diff;
  batchActions.innerHTML = '';
  if (patch.status === 'proposed') {
    const approve = actionButton('수정안 승인 및 복사본에 적용', 'primary');
    const reject = actionButton('수정안 거절', 'primary reject');
    approve.addEventListener('click', () => patchBatchAction(approve, 'approve'));
    reject.addEventListener('click', () => patchBatchAction(reject, 'reject'));
    batchActions.append(approve, reject);
  } else {
    const badge = document.createElement('span');
    badge.className = `badge ${patch.status === 'approved' ? 'validated' : 'rejected'}`;
    badge.textContent = patch.status === 'approved' ? '승인되어 복사본에 적용됨' : '수정안 거절됨';
    batchActions.append(badge);
  }
}

async function patchBatchAction(button, action) {
  button.disabled = true;
  const patch = analysisResult.patch_batch;
  try {
    const response = await fetch(
      `/api/jobs/${currentJob.job_id}/patches/${encodeURIComponent(patch.patch_id)}/${action}`,
      { method: 'POST' },
    );
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || '패치 처리에 실패했습니다.');
    analysisResult.patch_batch = payload;
    renderBatchPatch();
  } catch (error) {
    alert(error.message);
    button.disabled = false;
  }
}

for (const button of document.querySelectorAll('.filter')) {
  button.addEventListener('click', () => {
    document.querySelector('.filter.active')?.classList.remove('active');
    button.classList.add('active');
    activeFilter = button.dataset.filter;
    renderFindings();
  });
}

function actionButton(label, classes) {
  const button = document.createElement('button');
  button.className = classes;
  button.textContent = label;
  return button;
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>'"]/g, (character) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
  })[character]);
}

function formatBytes(value) {
  if (!value) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const index = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1);
  return `${(value / 1024 ** index).toFixed(index ? 1 : 0)} ${units[index]}`;
}
