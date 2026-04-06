const DEMO_BASE = "static/demo/";

function formatTime(seconds) {
  const value = Math.max(0, Number(seconds) || 0);
  const m = Math.floor(value / 60);
  const s = (value % 60).toFixed(1).padStart(4, "0");
  return `${m}:${s}`;
}

function resolveDemoPath(path) {
  if (!path) return "";
  if (path.startsWith("http")) return path;
  return `${DEMO_BASE}${path}`;
}

function prettifyTechnique(raw) {
  const text = String(raw || "").toLowerCase().trim();
  if (!text) return "unknown punch";
  const handMap = {
    l: "left",
    r: "right",
  };
  const typeMap = {
    straight: "straight",
    hook: "hook",
    uppercut: "uppercut",
  };
  const hand = handMap[text[0]] || "";
  const type = Object.keys(typeMap).find((key) => text.includes(key));
  if (hand && type) return `${hand} ${typeMap[type]}`;
  return text.replaceAll("_", " ").replaceAll("-", " ");
}

function prettifyEffect(raw) {
  const value = String(raw || "").trim().toLowerCase();
  if (!value) return "unknown";
  return value;
}

function prettifyFighterName(raw) {
  return String(raw || "")
    .replaceAll("-", " ")
    .replaceAll("'", "'")
    .trim();
}

function extractMatchupText(sample) {
  const source = String((sample && (sample.title || sample.source_title)) || "");
  const parts = source.split("_");
  if (parts.length < 3) return "";
  const blue = prettifyFighterName(parts[1]);
  const red = prettifyFighterName(parts.slice(2).join("_"));
  if (!blue || !red) return "";
  const clipId = String((sample && sample.id) || "").trim();
  const prefix = clipId ? `${clipId}. ` : "";
  return `${prefix}<span class="matchup-red">${red}</span> v.s. <span class="matchup-blue">${blue}</span>`;
}

function scrollNodeIntoCenter(container, node) {
  if (!node) return;
  const containerRect = container.getBoundingClientRect();
  const nodeRect = node.getBoundingClientRect();
  const containerCenter = containerRect.top + containerRect.height / 2;
  const nodeCenter = nodeRect.top + nodeRect.height / 2;
  const delta = nodeCenter - containerCenter;
  container.scrollTop = Math.max(0, container.scrollTop + delta);
}

function pickActiveTimedNode(nodes, currentTime) {
  let bestNode = null;
  let bestScore = Infinity;
  nodes.forEach((node) => {
    const start = Number(node.dataset.start);
    const end = Number(node.dataset.end);
    const mid = Number.isFinite(Number(node.dataset.time))
      ? Number(node.dataset.time)
      : (start + end) / 2;
    let score;
    if (currentTime >= start && currentTime <= end + 0.12) {
      score = Math.abs(mid - currentTime);
    } else if (currentTime < start) {
      score = (start - currentTime) + 1.5;
    } else {
      score = (currentTime - end) + 2.0;
    }
    if (score < bestScore) {
      bestScore = score;
      bestNode = node;
    }
  });
  if (!bestNode) return null;
  return bestScore <= 1.0 ? bestNode : null;
}

function pickActiveEventNode(container, currentTime) {
  return pickActiveTimedNode(Array.from(container.querySelectorAll(".event-item")), currentTime);
}

function pickActiveCommentaryNode(container, currentTime) {
  return pickActiveTimedNode(Array.from(container.querySelectorAll(".commentary-item")), currentTime);
}

function nearestSkeletonFrame(frames, time) {
  if (!frames || !frames.length) return null;
  let lo = 0;
  let hi = frames.length - 1;
  while (lo < hi) {
    const mid = Math.floor((lo + hi) / 2);
    if (frames[mid].time < time) lo = mid + 1;
    else hi = mid;
  }
  const current = frames[lo];
  const previous = frames[Math.max(0, lo - 1)];
  if (!current) return previous;
  return Math.abs(current.time - time) < Math.abs(previous.time - time) ? current : previous;
}

function buildEventItem(event, colorClass) {
  const item = document.createElement("article");
  item.className = `event-item ${colorClass}`;
  item.dataset.time = String(event.time);
  item.dataset.start = String(event.start_time !== undefined ? event.start_time : (event.time !== undefined ? event.time : 0));
  item.dataset.end = String(event.end_time !== undefined ? event.end_time : (event.time !== undefined ? event.time : 0));
  item.innerHTML = `
    <time>${formatTime(event.time)}</time>
    <strong>${prettifyTechnique(event.technique)}</strong>
    <span>${prettifyEffect(event.effect)} to ${event.target || "target"}</span>
  `;
  return item;
}

function buildCommentaryItem(commentary) {
  const item = document.createElement("article");
  item.className = `commentary-item ${commentary.class_name}`;
  item.dataset.start = String(commentary.start_time);
  item.dataset.end = String(commentary.end_time);
  item.innerHTML = `
    <header>
      <time>${formatTime(commentary.start_time)} - ${formatTime(commentary.end_time)}</time>
    </header>
    <div class="caption-text">${commentary.text}</div>
  `;
  return item;
}

function getVideoRenderRect(video, canvas) {
  const videoWidth = Number(video.videoWidth) || 0;
  const videoHeight = Number(video.videoHeight) || 0;
  const canvasWidth = Number(canvas.width) || 0;
  const canvasHeight = Number(canvas.height) || 0;
  if (!videoWidth || !videoHeight || !canvasWidth || !canvasHeight) {
    return { x: 0, y: 0, width: canvasWidth, height: canvasHeight };
  }
  const scale = Math.min(canvasWidth / videoWidth, canvasHeight / videoHeight);
  const width = videoWidth * scale;
  const height = videoHeight * scale;
  return {
    x: (canvasWidth - width) / 2,
    y: (canvasHeight - height) / 2,
    width,
    height,
  };
}

function drawSkeleton(ctx, frame, edges, color, renderRect) {
  const points = color === "red" ? frame.red : frame.blue;
  if (!points || !points.length) return;
  ctx.strokeStyle = color === "red" ? "#d43f36" : "#2f69c8";
  ctx.lineWidth = 2.5;
  ctx.lineCap = "round";
  edges.forEach(([a, b]) => {
    const pa = points[a];
    const pb = points[b];
    if (!pa || !pb) return;
    ctx.beginPath();
    ctx.moveTo(renderRect.x + pa[0] * renderRect.width, renderRect.y + pa[1] * renderRect.height);
    ctx.lineTo(renderRect.x + pb[0] * renderRect.width, renderRect.y + pb[1] * renderRect.height);
    ctx.stroke();
  });
}

document.addEventListener("DOMContentLoaded", async () => {
  const root = document.querySelector("[data-demo-root]");
  if (!root) return;

  const picker = root.querySelector("[data-sample-picker]");
  const video = root.querySelector("[data-demo-video]");
  const canvas = root.querySelector("[data-skeleton-canvas]");
  const toggle = root.querySelector("[data-toggle-skeleton]");
  const redList = root.querySelector("[data-red-events]");
  const blueList = root.querySelector("[data-blue-events]");
  const commentaryList = root.querySelector("[data-commentary]");
  const matchup = root.querySelector("[data-matchup]");
  const sampleTitle = root.querySelector("[data-sample-title]");
  const sampleMeta = root.querySelector("[data-sample-meta]");
  const sampleSummary = root.querySelector("[data-sample-summary]");

  const ctx = canvas.getContext("2d");
  let activeSample = null;
  let activeData = null;
  let rafId = null;
  let lastRedActiveKey = "";
  let lastBlueActiveKey = "";
  let lastCommentaryActiveKey = "";

  function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.round(rect.width));
    canvas.height = Math.max(1, Math.round(rect.height));
  }

  function syncActiveState(currentTime) {
    const redActiveNode = pickActiveEventNode(redList, currentTime);
    const blueActiveNode = pickActiveEventNode(blueList, currentTime);
    const commentaryActiveNode = pickActiveCommentaryNode(commentaryList, currentTime);
    redList.querySelectorAll(".event-item").forEach((node) => {
      node.classList.toggle("is-active", node === redActiveNode);
    });
    blueList.querySelectorAll(".event-item").forEach((node) => {
      node.classList.toggle("is-active", node === blueActiveNode);
    });
    commentaryList.querySelectorAll(".commentary-item").forEach((node) => {
      node.classList.toggle("is-active", node === commentaryActiveNode);
    });

    const redKey = redActiveNode ? `${redActiveNode.dataset.start}-${redActiveNode.dataset.end}` : "";
    const blueKey = blueActiveNode ? `${blueActiveNode.dataset.start}-${blueActiveNode.dataset.end}` : "";
    const commentaryKey = commentaryActiveNode ? `${commentaryActiveNode.dataset.start}-${commentaryActiveNode.dataset.end}` : "";
    if (redKey && redKey !== lastRedActiveKey) {
      lastRedActiveKey = redKey;
      scrollNodeIntoCenter(redList, redActiveNode);
    }
    if (blueKey && blueKey !== lastBlueActiveKey) {
      lastBlueActiveKey = blueKey;
      scrollNodeIntoCenter(blueList, blueActiveNode);
    }
    if (commentaryKey && commentaryKey !== lastCommentaryActiveKey) {
      lastCommentaryActiveKey = commentaryKey;
      scrollNodeIntoCenter(commentaryList, commentaryActiveNode);
    }
  }

  function renderOverlay() {
    if (!activeData) return;
    resizeCanvas();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!toggle.checked || !activeData.skeleton || !activeData.skeleton.available) return;
    const frame = nearestSkeletonFrame(activeData.skeleton.frames, video.currentTime || 0);
    if (!frame) return;
    const renderRect = getVideoRenderRect(video, canvas);
    drawSkeleton(ctx, frame, activeData.skeleton.edges || [], "red", renderRect);
    drawSkeleton(ctx, frame, activeData.skeleton.edges || [], "blue", renderRect);
  }

  function tick() {
    syncActiveState(video.currentTime || 0);
    renderOverlay();
    rafId = window.requestAnimationFrame(tick);
  }

  function populateTimeline(data) {
    redList.innerHTML = "";
    blueList.innerHTML = "";
    commentaryList.innerHTML = "";

    data.events.filter((ev) => ev.side === "red").forEach((ev) => redList.appendChild(buildEventItem(ev, "red")));
    data.events.filter((ev) => ev.side === "blue").forEach((ev) => blueList.appendChild(buildEventItem(ev, "blue")));
    data.commentary.forEach((row) => commentaryList.appendChild(buildCommentaryItem(row)));

    if (sampleTitle) sampleTitle.textContent = data.title;
    sampleMeta.innerHTML = `
      <span class="meta-pill">Clip ${formatTime(data.clip.start_time)} - ${formatTime(data.clip.end_time)}</span>
      <span class="meta-pill">${data.counts.red_events} red events</span>
      <span class="meta-pill">${data.counts.blue_events} blue events</span>
      <span class="meta-pill">${data.counts.commentary} commentary lines</span>
    `;
    if (sampleSummary) {
      sampleSummary.textContent =
        `A ${data.clip.duration.toFixed(0)}-second evaluation clip with synchronized punch events, commentary categories, and ${data.counts.skeleton_frames} pose frames from the HMR postprocess output.`;
    }
    if (matchup) {
      matchup.innerHTML = extractMatchupText(activeSample);
    }
    lastRedActiveKey = "";
    lastBlueActiveKey = "";
    lastCommentaryActiveKey = "";
    redList.scrollTop = 0;
    blueList.scrollTop = 0;
    commentaryList.scrollTop = 0;
  }

  async function loadSample(sample, chip) {
    if (rafId) {
      cancelAnimationFrame(rafId);
      rafId = null;
    }
    picker.querySelectorAll(".sample-chip").forEach((node) => node.classList.remove("is-active"));
    if (chip) chip.classList.add("is-active");
    activeSample = sample;

    const response = await fetch(resolveDemoPath(sample.data_src));
    activeData = await response.json();
    video.src = resolveDemoPath(sample.video_src);
    video.load();
    populateTimeline(activeData);
    renderOverlay();
    tick();
  }

  const manifestResponse = await fetch("static/demo/data/manifest.json");
  const manifest = await manifestResponse.json();
  const samples = manifest.samples || [];

  samples.forEach((sample, index) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "sample-chip";
    chip.textContent = `${sample.id}`;
    chip.addEventListener("click", () => loadSample(sample, chip));
    picker.appendChild(chip);
    if (index === 0) {
      loadSample(sample, chip);
    }
  });

  toggle.addEventListener("change", renderOverlay);
  window.addEventListener("resize", renderOverlay);
  video.addEventListener("seeked", renderOverlay);
  video.addEventListener("loadedmetadata", renderOverlay);
});
