// Web_Interface request flow + status indicator (task 16.1)
//
// Scope (16.1): upload form submission, POST multipart to the backend, and a
// processing status indicator shown while the request is in flight. Results
// rendering, category summary, download control, and error-message rendering
// are intentionally NOT implemented here (task 16.2).
//
// Functions are kept small, named, and exported so the DOM behavior can be
// tested later (task 16.3) and extended by task 16.2.

// Default backend endpoint. Same-origin by default; override by setting
// window.PII_GUARDRAIL_API_URL before this module runs, if needed.
export const DEFAULT_API_URL = "/api/redact";

import { runPaddleOcr, groupWordsIntoSentences, unionBoxes, getRedactionBoxes } from "./paddle_ocr.js";
export { getRedactionBoxes };

let currentOverlayData = null;
let lastProcessedImage = null;
let lastRawSegments = null;

/**
 * Draw or highlight bounding boxes on the preview canvas.
 */
export function drawBoundingBoxesOnCanvas(canvas, img, sentences, activeIndex = null, highlightCustomBox = null) {
  if (!canvas || !img) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);
  ctx.drawImage(img, 0, 0, w, h);

  if (Array.isArray(sentences)) {
    sentences.forEach((s, idx) => {
      const box = s && s.box ? s.box : null;
      if (!box) return;

      const isActive = activeIndex === idx;

      ctx.lineWidth = isActive ? 3 : 1.5;
      ctx.strokeStyle = isActive ? "#ef4444" : "#2563eb";
      ctx.fillStyle = isActive ? "rgba(239, 68, 68, 0.22)" : "rgba(37, 99, 235, 0.12)";

      ctx.fillRect(box.x, box.y, box.width, box.height);
      ctx.strokeRect(box.x, box.y, box.width, box.height);

      // Badge label (#1, #2)
      const label = `#${idx + 1}`;
      ctx.font = "bold 13px system-ui, sans-serif";
      const textMetrics = ctx.measureText(label);
      const badgeW = textMetrics.width + 8;
      const badgeH = 18;

      const badgeY = Math.max(0, box.y - badgeH);
      ctx.fillStyle = isActive ? "#ef4444" : "#2563eb";
      ctx.fillRect(box.x, badgeY, badgeW, badgeH);

      ctx.fillStyle = "#ffffff";
      ctx.fillText(label, box.x + 4, badgeY + 14);
    });
  }

  if (highlightCustomBox) {
    ctx.lineWidth = 3;
    ctx.strokeStyle = "#f59e0b";
    ctx.fillStyle = "rgba(245, 158, 11, 0.35)";
    ctx.fillRect(highlightCustomBox.x, highlightCustomBox.y, highlightCustomBox.width, highlightCustomBox.height);
    ctx.strokeRect(highlightCustomBox.x, highlightCustomBox.y, highlightCustomBox.width, highlightCustomBox.height);
  }
}

/**
 * Automatically scrolls the canvas wrapper smoothly to center the given box in view.
 */
export function scrollCanvasToBox(box) {
  if (!box || !currentOverlayData || !currentOverlayData.canvas) return;
  const canvas = currentOverlayData.canvas;
  const wrapper = canvas.parentElement; // .ocr-canvas-wrapper
  if (!wrapper || typeof wrapper.scrollTo !== "function") return;

  const scale = canvas.clientHeight && canvas.height ? canvas.clientHeight / canvas.height : 1;
  const boxTop = box.y * scale;
  const boxHeight = box.height * scale;
  const boxCenter = boxTop + boxHeight / 2;

  const wrapperHeight = wrapper.clientHeight;
  const targetScrollTop = boxCenter - wrapperHeight / 2;

  wrapper.scrollTo({
    top: Math.max(0, targetScrollTop),
    behavior: "smooth",
  });
}

/**
 * Highlight a specific bounding box by index (or remove highlight with null).
 */
export function highlightPreviewBox(index) {
  if (currentOverlayData && currentOverlayData.canvas && currentOverlayData.imgElement) {
    const items = currentOverlayData.items || currentOverlayData.sentences;
    drawBoundingBoxesOnCanvas(
      currentOverlayData.canvas,
      currentOverlayData.imgElement,
      items,
      index
    );
    if (index !== null && Array.isArray(items) && items[index] && items[index].box) {
      scrollCanvasToBox(items[index].box);
    }
  }
}

/**
 * Highlight a specific word-level box (in amber/gold) on the canvas.
 */
export function highlightWordBox(box) {
  if (currentOverlayData && currentOverlayData.canvas && currentOverlayData.imgElement) {
    drawBoundingBoxesOnCanvas(
      currentOverlayData.canvas,
      currentOverlayData.imgElement,
      currentOverlayData.items || currentOverlayData.sentences,
      null,
      box
    );
    if (box) {
      scrollCanvasToBox(box);
    }
  }
}

/**
 * Render visual canvas with bounding boxes into the result container.
 */
export function renderCanvasPreview(imgElement, sentences, container) {
  if (!container) return;
  container.replaceChildren();

  const figure = document.createElement("figure");
  figure.className = "ocr-preview-figure";

  const wrapper = document.createElement("div");
  wrapper.className = "ocr-canvas-wrapper";

  const canvas = document.createElement("canvas");
  canvas.id = "ocr-preview-canvas";
  canvas.width = imgElement.naturalWidth || imgElement.width;
  canvas.height = imgElement.naturalHeight || imgElement.height;

  wrapper.appendChild(canvas);
  figure.appendChild(wrapper);

  const caption = document.createElement("figcaption");
  caption.className = "ocr-preview-caption";
  caption.textContent = `Visual Sentence Bounding Boxes (${sentences.length} detected sentences)`;
  figure.appendChild(caption);

  container.appendChild(figure);

  currentOverlayData = {
    canvas,
    imgElement,
    sentences,
    items: sentences,
  };

  drawBoundingBoxesOnCanvas(canvas, imgElement, sentences, null);
}

/**
 * Resolve the backend endpoint URL. Prefers a runtime-configurable global,
 * falling back to the same-origin default.
 * @param {object} [globalObj] - object to read overrides from (defaults to globalThis).
 * @returns {string}
 */
export function getApiUrl(globalObj = globalThis) {
  const configured =
    globalObj && typeof globalObj.PII_GUARDRAIL_API_URL === "string"
      ? globalObj.PII_GUARDRAIL_API_URL.trim()
      : "";
  return configured || DEFAULT_API_URL;
}

/**
 * Build the multipart FormData body for a redact request.
 * @param {File|Blob} file - the selected image file.
 * @returns {FormData}
 */
export function buildFormData(file) {
  const formData = new FormData();
  // Backend expects the file under the `image` key (design: POST /api/redact).
  formData.append("image", file);
  return formData;
}

/**
 * Submit an image file to the backend and return the parsed JSON response.
 *
 * Resolves with the parsed body for successful (2xx) responses and rejects
 * with an Error for non-2xx responses or network/parse failures. The rejection
 * Error carries `status` and (when available) `payload` so task 16.2 can render
 * a detailed error message.
 *
 * @param {File|Blob} file - the image file to upload.
 * @param {object} [options]
 * @param {string} [options.apiUrl] - endpoint override (defaults to getApiUrl()).
 * @param {typeof fetch} [options.fetchImpl] - fetch implementation (for testing).
 * @returns {Promise<object>} parsed response body.
 */
export async function submitImage(file, options = {}) {
  if (!file) {
    throw new Error("No file selected.");
  }

  const apiUrl = options.apiUrl || getApiUrl();
  const fetchImpl = options.fetchImpl || globalThis.fetch;

  if (typeof fetchImpl !== "function") {
    throw new Error("fetch is not available in this environment.");
  }

  const response = await fetchImpl(apiUrl, {
    method: "POST",
    body: buildFormData(file),
  });

  // Attempt to parse JSON regardless of status; the backend returns JSON for
  // both success and error responses.
  let payload;
  try {
    payload = await response.json();
  } catch (parseErr) {
    payload = undefined;
  }

  if (!response.ok) {
    const message =
      payload && payload.error && payload.error.message
        ? payload.error.message
        : `Request failed with status ${response.status}.`;
    const error = new Error(message);
    error.status = response.status;
    error.payload = payload;
    throw error;
  }

  return payload;
}

/**
 * Show the processing status indicator (Requirement 9.5).
 * @param {HTMLElement|null} statusEl
 */
export function showStatus(statusEl) {
  if (!statusEl) return;
  statusEl.hidden = false;
  statusEl.setAttribute("aria-busy", "true");
}

/**
 * Hide the processing status indicator.
 * @param {HTMLElement|null} statusEl
 */
export function hideStatus(statusEl) {
  if (!statusEl) return;
  statusEl.hidden = true;
  statusEl.setAttribute("aria-busy", "false");
}

/**
 * Enable/disable the submit control while a request is in flight.
 * @param {HTMLButtonElement|null} submitBtn
 * @param {boolean} inFlight
 */
export function setInFlight(submitBtn, inFlight) {
  if (!submitBtn) return;
  submitBtn.disabled = inFlight;
}

/**
 * Resolve the key interactive elements from a root (document by default).
 * Centralized so ids stay consistent between HTML, this flow, and later tasks.
 * @param {Document|HTMLElement} [root]
 */
export function getElements(root = document) {
  return {
    form: root.getElementById("upload-form"),
    fileInput: root.getElementById("file-input"),
    submitBtn: root.getElementById("submit-btn"),
    statusIndicator: root.getElementById("status-indicator"),
    resultContainer: root.getElementById("result-container"),
    errorContainer: root.getElementById("error-container"),
    ocrDebug: root.getElementById("ocr-debug"),
    ocrDebugBody: root.getElementById("ocr-debug-body"),
    ocrDebugCount: root.getElementById("ocr-debug-count"),
  };
}

/**
 * Handle a form submission: show the status indicator, submit the image, and
 * hide the indicator once the request settles (success or failure).
 *
 * Result and error rendering are delegated to hooks so task 16.2 can supply the
 * concrete rendering without changing this flow. This function clears the
 * placeholder containers before each request.
 *
 * @param {Event} event - the submit event.
 * @param {object} elements - resolved elements from getElements().
 * @param {object} [hooks]
 * @param {(response: object) => void} [hooks.onResult] - called on success (16.2).
 * @param {(error: Error) => void} [hooks.onError] - called on failure (16.2).
 * @param {(file: File) => Promise<object>} [hooks.submit] - submit override (testing).
 * @returns {Promise<void>}
 */
export async function handleSubmit(event, elements, hooks = {}) {
  if (event && typeof event.preventDefault === "function") {
    event.preventDefault();
  }

  const {
    fileInput,
    submitBtn,
    statusIndicator,
    resultContainer,
    errorContainer,
    ocrDebug,
    ocrDebugBody,
    ocrDebugCount,
  } = elements || {};

  // Clear previous output placeholders.
  if (resultContainer) resultContainer.replaceChildren();
  if (errorContainer) errorContainer.replaceChildren();

  // Reset the OCR debug panel between requests.
  if (ocrDebugBody) ocrDebugBody.replaceChildren();
  if (ocrDebugCount) ocrDebugCount.textContent = "";
  if (ocrDebug) ocrDebug.hidden = true;

  const file = fileInput && fileInput.files ? fileInput.files[0] : null;
  if (!file) return;

  showStatus(statusIndicator);
  setInFlight(submitBtn, true);

  const statusTextEl = statusIndicator
    ? statusIndicator.querySelector(".status-text")
    : null;
  const setStatusText = (msg) => {
    if (statusTextEl) statusTextEl.textContent = msg;
  };

  try {
    if (hooks.submit) {
      // Injected submit (e.g. testing or custom hook)
      const response = await hooks.submit(file);
      if (typeof hooks.onResult === "function") {
        hooks.onResult(response);
      }
    } else {
      // Live browser environment: Run 100% In-Browser PaddleOCR!
      setStatusText("Initializing In-Browser PaddleOCR models...");

      const img = new Image();
      const imgUrl = URL.createObjectURL(file);
      img.src = imgUrl;
      await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
      });

      const sentences = await runPaddleOcr(img, {
        maxGapRatio: 1.2,
        onProgress: (msg) => setStatusText(msg),
      });

      lastProcessedImage = img;
      lastRawSegments = sentences.rawSegments || null;

      // Render visual canvas with bounding boxes
      renderCanvasPreview(img, sentences, resultContainer);

      // Render debug table focusing on recognized sentences and bounding boxes
      renderOcrDebug(sentences, elements);

      if (ocrDebug) {
        ocrDebug.open = true;
        ocrDebug.hidden = false;
      }
    }
  } catch (error) {
    if (typeof hooks.onError === "function") {
      hooks.onError(error);
    } else {
      renderError(error, elements);
    }
  } finally {
    hideStatus(statusIndicator);
    setInFlight(submitBtn, false);
    setStatusText("Processing...");
  }
}

/**
 * Build a data URL for the redacted image from the backend response payload.
 * @param {{format?: string, base64?: string}} redactedImage
 * @returns {string|null} the data URL, or null if data is missing.
 */
export function buildImageDataUrl(redactedImage) {
  if (!redactedImage || typeof redactedImage.base64 !== "string" || !redactedImage.base64) {
    return null;
  }
  const format = typeof redactedImage.format === "string" && redactedImage.format
    ? redactedImage.format
    : "png";
  return `data:image/${format};base64,${redactedImage.base64}`;
}

/**
 * Summarize the Sensitive_Category values present across the detected regions.
 * Returns an array of { category, count } sorted by descending count then name.
 * @param {Array<{categories?: string[]}>} regions
 * @returns {Array<{category: string, count: number}>}
 */
export function summarizeCategories(regions) {
  const counts = new Map();
  const list = Array.isArray(regions) ? regions : [];
  for (const region of list) {
    const categories = region && Array.isArray(region.categories) ? region.categories : [];
    for (const category of categories) {
      if (typeof category !== "string" || !category) continue;
      counts.set(category, (counts.get(category) || 0) + 1);
    }
  }
  return Array.from(counts.entries())
    .map(([category, count]) => ({ category, count }))
    .sort((a, b) => (b.count - a.count) || a.category.localeCompare(b.category));
}

/**
 * Render the raw OCR segments into the #ocr-debug panel.
 *
 * Shows every sentence / segment the OCR engine read (text, confidence, bounding box).
 * Focused specifically on recognized sentence text and exact bounding box coordinates.
 *
 * @param {Array<{text?: string, confidence?: number, box?: object}>} segments
 * @param {object} elements - resolved elements from getElements().
 */
export function renderOcrDebug(segments, elements) {
  const panel = elements && elements.ocrDebug;
  const body = elements && elements.ocrDebugBody;
  const countEl = elements && elements.ocrDebugCount;
  if (!panel || !body) return;

  body.replaceChildren();
  const list = Array.isArray(segments) ? segments : [];

  panel.hidden = false;

  // Flatten all word-level segments across all sentences
  const allWords = [];
  list.forEach((s) => {
    if (Array.isArray(s.words) && s.words.length > 0) {
      s.words.forEach((w) => {
        allWords.push({
          text: w.text,
          box: w.box,
          confidence: w.confidence !== undefined ? w.confidence : s.confidence,
        });
      });
    } else if (s && s.box) {
      allWords.push({
        text: s.text,
        box: s.box,
        confidence: s.confidence,
      });
    }
  });

  if (countEl) {
    countEl.textContent = `(${list.length} segments · ${allWords.length} words)`;
  }

  if (list.length === 0) {
    const empty = document.createElement("p");
    empty.className = "ocr-empty";
    empty.textContent = "OCR did not recognize any text in this image.";
    body.appendChild(empty);
    return;
  }

  // Detect if segments carry PII categories (for legacy test compatibility)
  const hasCategories = list.some(
    (s) => (Array.isArray(s.categories) && s.categories.length > 0) || s.sources
  );

  let viewMode = "sentences"; // "sentences" or "words"

  // Controls container (View toggle)
  const controlsTop = document.createElement("div");
  controlsTop.className = "ocr-controls-top";

  const toggleContainer = document.createElement("div");
  toggleContainer.className = "ocr-view-toggle";

  const btnSentences = document.createElement("button");
  btnSentences.type = "button";
  btnSentences.className = "btn-toggle active";
  btnSentences.textContent = `📄 Sentences View (${list.length})`;

  const btnWords = document.createElement("button");
  btnWords.type = "button";
  btnWords.className = "btn-toggle";
  btnWords.textContent = `🔤 Words View (${allWords.length})`;

  toggleContainer.appendChild(btnSentences);
  toggleContainer.appendChild(btnWords);
  controlsTop.appendChild(toggleContainer);

  const tableContainer = document.createElement("div");
  tableContainer.className = "ocr-table-container";

  function renderTable() {
    tableContainer.replaceChildren();

    if (viewMode === "sentences") {
      btnSentences.classList.add("active");
      btnWords.classList.remove("active");

      // Switch canvas overlay to sentence boxes
      if (currentOverlayData && currentOverlayData.canvas && currentOverlayData.imgElement) {
        currentOverlayData.items = list;
        drawBoundingBoxesOnCanvas(currentOverlayData.canvas, currentOverlayData.imgElement, list, null);
      }

      const table = document.createElement("table");
      table.className = "ocr-table";

      const thead = document.createElement("thead");
      const headRow = document.createElement("tr");

      const headings = hasCategories
        ? [
            "#",
            "Recognized sentence / text",
            "Classified as (by layer)",
            "Redacted?",
            "Confidence",
            "Box (x, y, w, h)",
          ]
        : [
            "#",
            "Recognized sentence / text",
            "Confidence",
            "Box (x, y, w, h)",
          ];

      for (const heading of headings) {
        const th = document.createElement("th");
        th.textContent = heading;
        headRow.appendChild(th);
      }
      thead.appendChild(headRow);
      table.appendChild(thead);

      const tbody = document.createElement("tbody");
      list.forEach((segment, index) => {
        const row = document.createElement("tr");
        row.dataset.index = String(index);

        const idxCell = document.createElement("td");
        idxCell.className = "ocr-index";
        idxCell.textContent = String(index + 1);
        row.appendChild(idxCell);

        const textCell = document.createElement("td");
        textCell.className = "ocr-text";
        const text = segment && typeof segment.text === "string" ? segment.text : "";
        textCell.textContent = text;
        row.appendChild(textCell);

        if (hasCategories) {
          const categories =
            segment && Array.isArray(segment.categories) ? segment.categories : [];
          const sources =
            segment && segment.sources && typeof segment.sources === "object"
              ? segment.sources
              : {};
          const sourceNames = Object.keys(sources);
          const catCell = document.createElement("td");
          catCell.className = "ocr-categories";

          if (categories.length === 0) {
            const none = document.createElement("span");
            none.className = "ocr-cat-none";
            none.textContent = "(not sensitive)";
            catCell.appendChild(none);
          } else if (sourceNames.length > 0) {
            for (const src of sourceNames.sort()) {
              const cats = Array.isArray(sources[src]) ? sources[src] : [];
              if (cats.length === 0) continue;
              const line = document.createElement("div");
              line.className = "ocr-source-line";

              const label = document.createElement("span");
              label.className = "ocr-source-label";
              label.textContent = `${src}:`;
              line.appendChild(label);

              for (const category of cats) {
                const tag = document.createElement("span");
                tag.className = "ocr-cat-tag";
                tag.textContent = String(category);
                line.appendChild(tag);
              }
              catCell.appendChild(line);
            }
          } else {
            for (const category of categories) {
              const tag = document.createElement("span");
              tag.className = "ocr-cat-tag";
              tag.textContent = String(category);
              catCell.appendChild(tag);
            }
          }
          row.appendChild(catCell);

          const redacted =
            segment && typeof segment.redacted === "boolean"
              ? segment.redacted
              : categories.length > 0;
          const redCell = document.createElement("td");
          redCell.className = redacted ? "ocr-redacted-yes" : "ocr-redacted-no";
          redCell.textContent = redacted ? "yes" : "no";
          row.appendChild(redCell);
        }

        const confCell = document.createElement("td");
        confCell.className = "ocr-confidence";
        const conf = segment && Number.isFinite(segment.confidence) ? segment.confidence : null;
        confCell.textContent = conf === null ? "-" : `${(conf * 100).toFixed(1)}%`;
        row.appendChild(confCell);

        const boxCell = document.createElement("td");
        boxCell.className = "ocr-box";
        const box = segment && segment.box ? segment.box : null;
        boxCell.textContent = box
          ? `${box.x}, ${box.y}, ${box.width}, ${box.height}`
          : "-";
        row.appendChild(boxCell);

        // Interactive box highlight on hover
        row.addEventListener("mouseenter", () => {
          row.classList.add("active-row");
          highlightPreviewBox(index);
        });
        row.addEventListener("mouseleave", () => {
          row.classList.remove("active-row");
          highlightPreviewBox(null);
        });

        tbody.appendChild(row);
      });
      table.appendChild(tbody);
      tableContainer.appendChild(table);
    } else {
      // WORDS ONLY VIEW
      btnWords.classList.add("active");
      btnSentences.classList.remove("active");

      // Switch canvas overlay to show all word boxes
      if (currentOverlayData && currentOverlayData.canvas && currentOverlayData.imgElement) {
        currentOverlayData.items = allWords;
        drawBoundingBoxesOnCanvas(currentOverlayData.canvas, currentOverlayData.imgElement, allWords, null);
      }

      const table = document.createElement("table");
      table.className = "ocr-table";

      const thead = document.createElement("thead");
      const headRow = document.createElement("tr");
      ["#", "Recognized Word", "Confidence", "Word Box (x, y, w, h)"].forEach((h) => {
        const th = document.createElement("th");
        th.textContent = h;
        headRow.appendChild(th);
      });
      thead.appendChild(headRow);
      table.appendChild(thead);

      const tbody = document.createElement("tbody");
      allWords.forEach((word, idx) => {
        const row = document.createElement("tr");
        row.dataset.index = String(idx);

        const idxCell = document.createElement("td");
        idxCell.className = "ocr-index";
        idxCell.textContent = String(idx + 1);
        row.appendChild(idxCell);

        const textCell = document.createElement("td");
        textCell.className = "ocr-text";
        textCell.style.fontWeight = "600";
        textCell.textContent = word.text;
        row.appendChild(textCell);

        const confCell = document.createElement("td");
        confCell.className = "ocr-confidence";
        const conf = word && Number.isFinite(word.confidence) ? word.confidence : null;
        confCell.textContent = conf === null ? "-" : `${(conf * 100).toFixed(1)}%`;
        row.appendChild(confCell);

        const boxCell = document.createElement("td");
        boxCell.className = "ocr-box";
        const box = word && word.box ? word.box : null;
        boxCell.textContent = box
          ? `${box.x}, ${box.y}, ${box.width}, ${box.height}`
          : "-";
        row.appendChild(boxCell);

        row.addEventListener("mouseenter", () => {
          row.classList.add("active-row");
          highlightPreviewBox(idx);
        });
        row.addEventListener("mouseleave", () => {
          row.classList.remove("active-row");
          highlightPreviewBox(null);
        });

        tbody.appendChild(row);
      });
      table.appendChild(tbody);
      tableContainer.appendChild(table);
    }
  }

  btnSentences.addEventListener("click", () => {
    viewMode = "sentences";
    renderTable();
  });

  btnWords.addEventListener("click", () => {
    viewMode = "words";
    renderTable();
  });

  body.appendChild(controlsTop);
  body.appendChild(tableContainer);

  renderTable();

  // Add Dual Copy JSON Action Buttons
  const actionContainer = document.createElement("div");
  actionContainer.className = "ocr-actions-bar";

  const copySentencesBtn = document.createElement("button");
  copySentencesBtn.type = "button";
  copySentencesBtn.className = "btn-secondary";
  copySentencesBtn.textContent = "📋 Copy Sentences + Words JSON";
  copySentencesBtn.onclick = () => {
    const jsonStr = JSON.stringify(list, null, 2);
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(jsonStr).then(() => {
        copySentencesBtn.textContent = "✅ Copied Sentences JSON!";
        setTimeout(() => (copySentencesBtn.textContent = "📋 Copy Sentences + Words JSON"), 2000);
      });
    }
  };
  actionContainer.appendChild(copySentencesBtn);

  const copyWordsBtn = document.createElement("button");
  copyWordsBtn.type = "button";
  copyWordsBtn.className = "btn-secondary";
  copyWordsBtn.textContent = "📋 Copy Words Only JSON";
  copyWordsBtn.onclick = () => {
    const jsonStr = JSON.stringify(allWords, null, 2);
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(jsonStr).then(() => {
        copyWordsBtn.textContent = "✅ Copied Words JSON!";
        setTimeout(() => (copyWordsBtn.textContent = "📋 Copy Words Only JSON"), 2000);
      });
    }
  };
  actionContainer.appendChild(copyWordsBtn);

  body.appendChild(actionContainer);
}

/**
 * Render the successful redaction response into #result-container (Req 9.2-9.4).
 *
 * Builds the DOM safely with createElement/textContent (no innerHTML with
 * response data) so untrusted content is never interpreted as markup.
 *
 * @param {object} response - parsed success payload from the backend.
 * @param {object} elements - resolved elements from getElements().
 */
export function renderResult(response, elements) {
  const container = elements && elements.resultContainer;
  if (!container) return;
  container.replaceChildren();

  const data = response || {};
  const detection = data.detection_result || {};

  // --- OCR debug view: raw recognized text (independent of the result box) ---
  renderOcrDebug(data.ocr_segments, elements);

  // --- Redacted image (Requirement 9.2) ---
  const dataUrl = buildImageDataUrl(data.redacted_image);
  if (dataUrl) {
    const figure = document.createElement("figure");
    figure.className = "result-figure";

    const img = document.createElement("img");
    img.id = "redacted-image";
    img.className = "redacted-image";
    img.alt = "Redacted image";
    img.src = dataUrl;
    figure.appendChild(img);
    container.appendChild(figure);

    // --- Download control (Requirement 9.4) ---
    const format =
      data.redacted_image && typeof data.redacted_image.format === "string" && data.redacted_image.format
        ? data.redacted_image.format
        : "png";
    const download = document.createElement("a");
    download.id = "download-link";
    download.className = "download-link";
    download.href = dataUrl;
    download.download = `redacted.${format}`;
    download.textContent = "Download redacted image";
    container.appendChild(download);
  }

  // --- Region count + category summary (Requirement 9.3) ---
  const count = Number.isFinite(detection.count) ? detection.count : 0;

  const countEl = document.createElement("p");
  countEl.id = "region-count";
  countEl.className = "region-count";
  const label = count === 1 ? "sensitive region" : "sensitive regions";
  countEl.textContent = `${count} ${label} detected`;
  container.appendChild(countEl);

  const summary = summarizeCategories(detection.regions);
  const summaryEl = document.createElement("ul");
  summaryEl.id = "category-summary";
  summaryEl.className = "category-summary";
  if (summary.length === 0) {
    const item = document.createElement("li");
    item.className = "category-item category-empty";
    item.textContent = "No sensitive categories detected";
    summaryEl.appendChild(item);
  } else {
    for (const { category, count: catCount } of summary) {
      const item = document.createElement("li");
      item.className = "category-item";
      item.textContent = `${category} (${catCount})`;
      summaryEl.appendChild(item);
    }
  }
  container.appendChild(summaryEl);

  // --- Warnings shown alongside successful results (design.md, Req 9.6) ---
  const warnings = Array.isArray(data.warnings) ? data.warnings : [];
  const combinedWarnings = warnings.slice();
  if (data.quality_sufficient === false) {
    combinedWarnings.push(
      "Image quality was insufficient for reliable detection; results may be incomplete."
    );
  }
  if (combinedWarnings.length > 0) {
    const warnEl = document.createElement("div");
    warnEl.id = "warnings";
    warnEl.className = "warnings";
    warnEl.setAttribute("role", "status");
    for (const warning of combinedWarnings) {
      if (typeof warning !== "string" || !warning) continue;
      const p = document.createElement("p");
      p.className = "warning";
      p.textContent = warning;
      warnEl.appendChild(p);
    }
    if (warnEl.childElementCount > 0) {
      container.appendChild(warnEl);
    }
  }
}

/**
 * Render a human-readable error into #error-container (Requirement 9.6).
 *
 * Uses the Error's `.message` plus optional structured detail/code carried on
 * `.payload.error`. DOM is built safely (textContent, no innerHTML).
 *
 * @param {Error & {payload?: object, status?: number}} error
 * @param {object} elements - resolved elements from getElements().
 */
export function renderError(error, elements) {
  const container = elements && elements.errorContainer;
  if (!container) return;
  container.replaceChildren();

  const message =
    (error && typeof error.message === "string" && error.message) ||
    "An unexpected error occurred.";
  const structured = error && error.payload && error.payload.error ? error.payload.error : {};

  const messageEl = document.createElement("p");
  messageEl.id = "error-message";
  messageEl.className = "error-message";
  messageEl.textContent = message;
  container.appendChild(messageEl);

  if (structured && typeof structured.detail === "string" && structured.detail) {
    const detailEl = document.createElement("p");
    detailEl.id = "error-detail";
    detailEl.className = "error-detail";
    detailEl.textContent = structured.detail;
    container.appendChild(detailEl);
  }

  if (structured && typeof structured.code === "string" && structured.code) {
    const codeEl = document.createElement("p");
    codeEl.id = "error-code";
    codeEl.className = "error-code";
    codeEl.textContent = `Error code: ${structured.code}`;
    container.appendChild(codeEl);
  }
}

/**
 * Wire the upload form to the submit flow. Safe to call once the DOM is ready.
 * @param {object} [hooks] - forwarded to handleSubmit (result/error rendering).
 * @param {Document|HTMLElement} [root]
 * @returns {object} the resolved elements.
 */
export function init(hooks = {}, root = document) {
  const elements = getElements(root);
  if (elements.form) {
    elements.form.addEventListener("submit", (event) =>
      handleSubmit(event, elements, hooks)
    );
  }
  hideStatus(elements.statusIndicator);
  return elements;
}

/**
 * Build the default result/error rendering hooks bound to the given elements.
 * Kept separate so the default browser flow and tests share the same wiring
 * while renderResult/renderError remain independently testable.
 * @param {object} elements - resolved elements from getElements().
 * @returns {{onResult: (resp: object) => void, onError: (err: Error) => void}}
 */
export function createRenderHooks(elements) {
  return {
    onResult: (response) => renderResult(response, elements),
    onError: (error) => renderError(error, elements),
  };
}

/**
 * Initialize the default browser flow: resolve elements and wire the render
 * hooks so successful results and errors are drawn into the page.
 * @param {Document|HTMLElement} [root]
 * @returns {object} the resolved elements.
 */
export function initDefault(root = document) {
  const elements = getElements(root);
  return init(createRenderHooks(elements), root);
}

// Auto-initialize in a browser context (skipped under test/module import where
// there is no document).
if (typeof document !== "undefined" && typeof window !== "undefined") {
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => initDefault());
  } else {
    initDefault();
  }
}
