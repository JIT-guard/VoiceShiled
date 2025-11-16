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
const thresholdInput = document.getElementById("threshold-input");
const thresholdValue = document.getElementById("threshold-value");

if (thresholdInput && thresholdValue) {
  const updateThresholdLabel = () => {
    thresholdValue.textContent = Number.parseFloat(thresholdInput.value).toFixed(2);
  };
  thresholdInput.addEventListener("input", updateThresholdLabel);
  updateThresholdLabel();
}

form?.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!form.file.files.length) {
    alert("Pick an audio or video file first.");
    return;
  }

  statusPanel.hidden = false;
  metadataBlock.hidden = true;
  segmentsContainer.hidden = true;
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
  segmentsContainer.hidden = false;

  inputKindEl.textContent = metadata.input_kind;
  modelSizeEl.textContent = metadata.whisper_model;
  censoredCountEl.textContent = metadata.censored_segments;

  downloadLink.hidden = false;
  downloadLink.href = payload.downloadUrl;
  downloadLink.download = metadata.download_filename;

  renderSegments(metadata.segments || []);
}

function renderSegments(segments) {
  segmentList.innerHTML = "";
  if (!segments.length) {
    const empty = document.createElement("p");
    empty.textContent = "No transcript segments were returned.";
    segmentsContainer.innerHTML = "";
    segmentsContainer.append(empty);
    return;
  }
  segmentsContainer.innerHTML = "<h3>Transcript</h3>";
  segmentsContainer.append(segmentList);

  segments.forEach((segment) => {
    const item = document.createElement("li");
    if (segment.censored) {
      item.classList.add("flagged");
    }
    const timing = document.createElement("strong");
    timing.textContent = `${formatTime(segment.start)} → ${formatTime(segment.end)}`;
    item.append(timing);
    const content = document.createElement("span");
    content.textContent = segment.text || "—";
    item.append(content);
    segmentList.append(item);
  });
}

function formatTime(seconds = 0) {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60)
    .toString()
    .padStart(2, "0");
  return `${mins}:${secs}`;
}
