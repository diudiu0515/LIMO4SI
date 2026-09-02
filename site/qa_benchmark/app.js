const data = window.QA_DATA;
let activeGroup = 0;
let activeView = 'original';

const groupTabs = document.getElementById('groupTabs');
const caseTitle = document.getElementById('caseTitle');
const mainImage = document.getElementById('mainImage');
const mainVideo = document.getElementById('mainVideo');
const mainVideoSource = document.getElementById('mainVideoSource');
const videoFallback = document.getElementById('videoFallback');
const qaList = document.getElementById('qaList');
const taskGrid = document.getElementById('taskGrid');
const videoBtn = document.getElementById('videoBtn');
const originalBtn = document.getElementById('originalBtn');
const topdownBtn = document.getElementById('topdownBtn');
const groupJson = document.getElementById('groupJson');
const toggleGroupJson = document.getElementById('toggleGroupJson');

function fmtDistance(value) {
  return typeof value === 'number' ? `${value.toFixed(2)} m` : 'n/a';
}

function renderTabs() {
  groupTabs.innerHTML = '';
  data.groups.forEach((group, index) => {
    const okCount = group.qa.filter(item => item.status === 'ok').length;
    const button = document.createElement('button');
    button.className = `tab ${index === activeGroup ? 'active' : ''}`;
    button.type = 'button';
    button.innerHTML = `<strong>Case ${index + 1}</strong><span>${group.name} · ${okCount}/${group.qa.length} answered</span>`;
    button.addEventListener('click', () => {
      activeGroup = index;
      render();
    });
    groupTabs.appendChild(button);
  });
}

function renderTaskGrid() {
  if (!taskGrid || !Array.isArray(data.tasks)) return;
  taskGrid.innerHTML = '';
  data.tasks.forEach((task, index) => {
    const card = document.createElement('article');
    card.className = 'taskCard';
    card.innerHTML = `
      <div class="taskIndex">${index + 1}</div>
      <div>
        <h4>${escapeHtml(task.name)}</h4>
        <p>${escapeHtml(task.description)}</p>
      </div>
    `;
    taskGrid.appendChild(card);
  });
}

function renderQA(group) {
  qaList.innerHTML = '';
  group.qa.forEach((item, index) => {
    const card = document.createElement('article');
    card.className = `card ${item.task_id || ''}`;
    const statusClass = item.status === 'ok' ? 'ok' : 'skip';
    const options = Array.isArray(item.options) ? item.options : [];
    const optionsHtml = options.length ? `
      <div class="optionsGrid">
        ${options.map(opt => `
          <div class="option ${opt.label === item.correct_option ? 'correct' : ''}">
            <span class="optionLabel">${escapeHtml(opt.label)}</span>
            <span class="optionText">${escapeHtml(opt.text)}</span>
          </div>
        `).join('')}
      </div>
    ` : '';
    card.innerHTML = `
      <div class="cardHead">
        <div>
          <div class="taskName">${escapeHtml(item.task_name || item.task_id || 'Task')}</div>
          <div class="questionType">${escapeHtml(item.question_type || 'general')}</div>
          <div class="question">Q${index + 1}. ${escapeHtml(item.question || item.query || 'Untitled question')}</div>
        </div>
        <span class="pill ${statusClass}">${escapeHtml(item.status || 'unknown')}</span>
      </div>
      ${optionsHtml}
      <div class="answerBox">
        <div class="answerLabel">Correct answer: ${escapeHtml(item.correct_option || '')}</div>
        <div class="answer">${escapeHtml(item.correct_answer || item.answer || '')}</div>
        <div class="explanation">${formatAnswer(item.explanation || item.answer)}</div>
      </div>
      <div class="methodBox">${escapeHtml(item.method || '')}</div>
      <details class="lazyJson" data-json-kind="result" data-group-index="${activeGroup}" data-qa-index="${index}">
        <summary>Show computed result JSON</summary>
        <pre class="jsonBlock">Click to load JSON.</pre>
      </details>
      <details class="lazyJson" data-json-kind="raw" data-group-index="${activeGroup}" data-qa-index="${index}">
        <summary>Show raw input JSON</summary>
        <pre class="jsonBlock">Click to load JSON.</pre>
      </details>
    `;
    qaList.appendChild(card);
  });
}


function formatAnswer(answer) {
  if (typeof answer === 'string') return escapeHtml(answer);
  if (answer && typeof answer === 'object') {
    return `<pre class="inlineJson">${escapeHtml(JSON.stringify(answer, null, 2))}</pre>`;
  }
  return 'No answer recorded.';
}

function escapeHtml(text) {
  return String(text)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;');
}

function render() {
  const group = data.groups[activeGroup];
  renderTabs();
  renderTaskGrid();
  caseTitle.textContent = group.title;
  const showingVideo = activeView === 'video' && group.video_clip;
  mainVideo.classList.toggle('hidden', !showingVideo);
  mainImage.classList.toggle('hidden', showingVideo);
  videoFallback.classList.toggle('hidden', !showingVideo);
  if (showingVideo) {
    const clip = group.video_clip;
    const poster = group.original_image;
    if (mainVideoSource.getAttribute('src') !== clip) {
      mainVideo.pause();
      mainVideoSource.setAttribute('src', clip);
      mainVideo.setAttribute('poster', poster);
      mainVideo.load();
    }
    videoFallback.innerHTML = `If the video does not play, <a href="${escapeHtml(clip)}" target="_blank" rel="noreferrer">open the MP4 directly</a>.`;
  } else {
    mainVideo.pause();
    mainVideoSource.setAttribute('src', '');
    mainVideo.removeAttribute('poster');
    mainImage.src = activeView === 'topdown' ? group.topdown_image : group.original_image;
    mainImage.alt = `${group.name} ${activeView} evidence`;
  }
  videoBtn.classList.toggle('active', activeView === 'video');
  originalBtn.classList.toggle('active', activeView === 'original');
  topdownBtn.classList.toggle('active', activeView === 'topdown');
  renderQA(group);
  groupJson.textContent = 'Click Show group JSON to load.';
  groupJson.dataset.loaded = '';
  groupJson.classList.add('hidden');
  toggleGroupJson.textContent = 'Show group JSON';
}

videoBtn.addEventListener('click', () => {
  activeView = 'video';
  render();
});

originalBtn.addEventListener('click', () => {
  activeView = 'original';
  render();
});

topdownBtn.addEventListener('click', () => {
  activeView = 'topdown';
  render();
});

toggleGroupJson.addEventListener('click', () => {
  const isHidden = groupJson.classList.toggle('hidden');
  if (!isHidden && groupJson.dataset.loaded !== '1') {
    const group = data.groups[activeGroup];
    groupJson.textContent = JSON.stringify(group.raw_summary || group.dynamic_timeline || group, null, 2);
    groupJson.dataset.loaded = '1';
  }
  toggleGroupJson.textContent = isHidden ? 'Show group JSON' : 'Hide group JSON';
});

qaList.addEventListener('toggle', (event) => {
  const node = event.target;
  if (!node.classList || !node.classList.contains('lazyJson') || !node.open || node.dataset.loaded === '1') return;
  const group = data.groups[Number(node.dataset.groupIndex)];
  const item = group.qa[Number(node.dataset.qaIndex)];
  if (node.dataset.jsonKind === 'raw') {
    node.querySelector('pre').textContent = item.raw_json ? JSON.stringify(item.raw_json, null, 2) : `Raw source JSON is stored on disk at: ${item.raw_json_path || group.summary_path || 'unknown'}`;
  } else {
    node.querySelector('pre').textContent = JSON.stringify(item.result_json || {}, null, 2);
  }
  node.dataset.loaded = '1';
}, true);

render();
