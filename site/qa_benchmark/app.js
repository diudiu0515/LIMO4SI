const data = window.QA_DATA;
let activeGroup = 0;
let activeView = 'video';

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
    card.innerHTML = `
      <div class="cardHead">
        <div>
          <div class="taskName">${escapeHtml(item.task_name || item.task_id || 'Task')}</div>
          <div class="questionType">${escapeHtml(item.question_type || 'general')}</div>
          <div class="question">Q${index + 1}. ${escapeHtml(item.question || item.query || 'Untitled question')}</div>
          <div class="answer">${formatAnswer(item.answer)}</div>
        </div>
        <span class="pill ${statusClass}">${escapeHtml(item.status || 'unknown')}</span>
      </div>
      <div class="methodBox">${escapeHtml(item.method || '')}</div>
      <details>
        <summary>Show computed result JSON</summary>
        <pre class="jsonBlock">${escapeHtml(JSON.stringify(item.result_json || item.raw_json, null, 2))}</pre>
      </details>
      <details>
        <summary>Show raw input JSON</summary>
        <pre class="jsonBlock">${escapeHtml(JSON.stringify(item.raw_json, null, 2))}</pre>
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
  groupJson.textContent = JSON.stringify(group.raw_summary, null, 2);
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
  toggleGroupJson.textContent = isHidden ? 'Show group JSON' : 'Hide group JSON';
});

render();
