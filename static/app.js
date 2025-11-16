const form = document.getElementById("upload-form");
const statusPanel = document.getElementById("status-panel");
const statusMessage = document.getElementById("status-message");
const metadataBlock = document.getElementById("metadata");
const inputKindEl = document.getElementById("input-kind");
const modelSizeEl = document.getElementById("model-size");
const censoredCountEl = document.getElementById("censored-count");
const downloadLink = document.getElementById("download-link");
const segmentsContainer = document.getElementById("segments-container");
const segmentList = document.getElementById("segment-list");
const flaggedContainer = document.getElementById("flagged-container");
const flaggedList = document.getElementById("flagged-list");
const thresholdInput = document.getElementById("threshold-input");
const thresholdValue = document.getElementById("threshold-value");
const playerContainer = document.getElementById("player-container");
const videoPlayer = document.getElementById("video-player");
const audioPlayer = document.getElementById("audio-player");

if (thresholdInput && thresholdValue) {
  const updateThresholdLabel = () => {
    thresholdValue.textContent = Number.parseFloat(thresholdInput.value).toFixed(2);
  };
  thresholdInput.addEventListener("input", updateThresholdLabel);
  updateThresholdLabel();
}

segmentList?.addEventListener("click", handleBlurToggle);
flaggedList?.addEventListener("click", handleBlurToggle);

form?.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!form.file.files.length) {
    alert("Pick an audio or video file first.");
    return;
  }

  statusPanel.hidden = false;
  metadataBlock.hidden = true;
  segmentsContainer.hidden = true;
  if (flaggedContainer) {
    flaggedContainer.hidden = true;
  }
  resetPlayer();
  statusMessage.textContent = "Processing media, this can take a while for long files…";

  const submitButton = form.querySelector("button");
  if (submitButton) {
    submitButton.disabled = true;
  }

  try {
    const payload = new FormData(form);
    const response = await fetch("/api/process", {
      method: "POST",
      body: payload,
    });

    if (!response.ok) {
      const errorBody = await response.json().catch(() => ({}));
      throw new Error(errorBody.detail || "Processing failed");
    }

    const body = await response.json();
    populateResult(body);
  } catch (error) {
    console.error(error);
    statusMessage.textContent = error.message;
  } finally {
    if (submitButton) {
      submitButton.disabled = false;
    }
  }
});

function populateResult(payload) {
  const metadata = payload.metadata;
  statusMessage.textContent = "Completed";
  metadataBlock.hidden = false;

  inputKindEl.textContent = metadata.input_kind;
  modelSizeEl.textContent = metadata.whisper_model;
  censoredCountEl.textContent = metadata.censored_segments;

  downloadLink.hidden = false;
  downloadLink.href = payload.downloadUrl;
  downloadLink.download = metadata.download_filename;

  const inlineUrl = `${payload.downloadUrl}?inline=1`;
  renderPreview(metadata, inlineUrl);
  renderSegments(metadata.segments || []);
}

function renderSegments(segments) {
  segmentList.innerHTML = "";
  if (flaggedList) {
    flaggedList.innerHTML = "";
  }

  const flaggedSegments = segments.filter((segment) => segment.censored);
  if (flaggedContainer) {
    flaggedContainer.hidden = flaggedSegments.length === 0;
  }
  if (flaggedList) {
    if (!flaggedSegments.length) {
      const placeholder = document.createElement("li");
      placeholder.textContent = "No toxic language detected.";
      placeholder.classList.add("muted");
      flaggedList.append(placeholder);
    } else {
      flaggedSegments.forEach((segment) => {
        flaggedList.append(buildFlaggedItem(segment));
      });
    }
  }

  if (!segments.length) {
    const empty = document.createElement("li");
    empty.textContent = "No transcript segments were returned.";
    empty.classList.add("muted");
    segmentList.append(empty);
    segmentsContainer.hidden = false;
    return;
  }

  segments.forEach((segment) => {
    segmentList.append(buildTranscriptItem(segment));
  });
  segmentsContainer.hidden = false;
}

function formatTime(seconds = 0) {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60)
    .toString()
    .padStart(2, "0");
  return `${mins}:${secs}`;
}

function renderPreview(metadata, streamUrl) {
  resetPlayer();
  if (!playerContainer || !streamUrl) {
    return;
  }
  playerContainer.hidden = false;
  if (metadata.input_kind === "video") {
    if (videoPlayer) {
      videoPlayer.hidden = false;
      videoPlayer.src = streamUrl;
      videoPlayer.load();
    }
    if (audioPlayer) {
      audioPlayer.hidden = true;
    }
  } else {
    if (audioPlayer) {
      audioPlayer.hidden = false;
      audioPlayer.src = streamUrl;
      audioPlayer.load();
    }
    if (videoPlayer) {
      videoPlayer.hidden = true;
    }
  }
}

function resetPlayer() {
  if (videoPlayer) {
    videoPlayer.pause();
    videoPlayer.removeAttribute("src");
    videoPlayer.load();
    videoPlayer.hidden = true;
  }
  if (audioPlayer) {
    audioPlayer.pause();
    audioPlayer.removeAttribute("src");
    audioPlayer.load();
    audioPlayer.hidden = true;
  }
  if (playerContainer) {
    playerContainer.hidden = true;
  }
}

function buildTranscriptItem(segment) {
  const item = document.createElement("li");
  if (segment.censored) {
    item.classList.add("flagged");
  }
  const timing = document.createElement("strong");
  timing.textContent = `${formatTime(segment.start)} → ${formatTime(segment.end)}`;
  item.append(timing);

  const badge = document.createElement("span");
  badge.classList.add("badge");
  badge.textContent = `${Math.round((segment.score ?? 0) * 100)}%`;
  if (segment.censored) {
    badge.classList.add("flagged");
    badge.textContent = `Toxic ${badge.textContent}`;
  } else {
    badge.textContent = `Clean ${badge.textContent}`;
  }
  item.append(badge);

  const content = document.createElement("span");
  content.textContent = segment.text || "—";
  if (segment.censored) {
    content.classList.add("blur-text");
  }
  item.append(content);
  return item;
}

function buildFlaggedItem(segment) {
  const item = document.createElement("li");
  const timing = document.createElement("strong");
  timing.textContent = `${formatTime(segment.start)} → ${formatTime(segment.end)} · ${Math.round(
    (segment.score ?? 0) * 100,
  )}%`;
  item.append(timing);
  const text = document.createElement("span");
  text.textContent = segment.text || "—";
  text.classList.add("blur-text");
  item.append(text);
  return item;
}

function handleBlurToggle(event) {
  const target = event.target.closest(".blur-text");
  if (!target) {
    return;
  }
  target.classList.toggle("revealed");
}
