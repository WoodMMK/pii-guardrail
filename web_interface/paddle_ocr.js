/**
 * In-Browser PaddleOCR Engine using ONNX Runtime Web.
 *
 * 100% Client-side execution (Zero external API calls, private and local).
 * Models:
 *  - Detection: PP-OCRv3 Mobile (det.onnx ~2.4MB)
 *  - Recognition: PP-OCRv5 Mobile Thai (rec.onnx ~7.8MB)
 *  - Dictionary: Thai character list (dict.txt ~1.7KB)
 */

let detSession = null;
let recSession = null;
let characterDict = null;
let isInitializing = false;
let initPromise = null;

/**
 * Initialize ONNX Runtime Web and load the PaddleOCR models.
 * @param {object} [options]
 * @param {string} [options.vendorPath] - Path to WASM binaries (defaults to '/vendor/')
 * @param {string} [options.modelsPath] - Path to models (defaults to '/models/')
 * @returns {Promise<{detSession: object, recSession: object}>}
 */
export async function initPaddleOcr(options = {}) {
  if (detSession && recSession && characterDict) {
    return { detSession, recSession };
  }
  if (isInitializing && initPromise) {
    return initPromise;
  }

  isInitializing = true;
  initPromise = (async () => {
    const vendorPath = options.vendorPath || "/vendor/";
    const modelsPath = options.modelsPath || "/models/";

    // Configure ONNX Runtime Web WASM path
    if (globalThis.ort && globalThis.ort.env && globalThis.ort.env.wasm) {
      const normalizedVendor = vendorPath.endsWith("/") ? vendorPath : `${vendorPath}/`;
      globalThis.ort.env.wasm.wasmPaths = {
        mjs: `${normalizedVendor}ort-wasm-simd-threaded.mjs`,
        wasm: `${normalizedVendor}ort-wasm-simd-threaded.wasm`,
      };
      // Multi-threading requires SharedArrayBuffer (crossOriginIsolated)
      const supportsThreading = typeof crossOriginIsolated !== "undefined" && crossOriginIsolated;
      globalThis.ort.env.wasm.numThreads = supportsThreading
        ? Math.min(4, navigator.hardwareConcurrency || 2)
        : 1;
    }

    if (!globalThis.ort) {
      throw new Error(
        "ONNX Runtime Web (ort) is not loaded. Please ensure /vendor/ort.min.js is included."
      );
    }

    // 1. Fetch character dictionary for Thai model
    const dictResp = await fetch(`${modelsPath}languages/thai/dict.txt`);
    if (!dictResp.ok) {
      throw new Error(`Failed to load Thai dictionary: ${dictResp.statusText}`);
    }
    const dictText = await dictResp.text();
    const rawLines = dictText.split(/\r?\n/).filter((l) => l.length > 0);
    // PaddleOCR CTC: 0 is 'blank', followed by dictionary entries, and ending with ' ' (space)
    characterDict = ["blank", ...rawLines, " "];

    // 2. Create Detection & Recognition inference sessions
    const sessionOptions = {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    };

    [detSession, recSession] = await Promise.all([
      globalThis.ort.InferenceSession.create(
        `${modelsPath}detection/v3/det.onnx`,
        sessionOptions
      ),
      globalThis.ort.InferenceSession.create(
        `${modelsPath}languages/thai/rec.onnx`,
        sessionOptions
      ),
    ]);

    isInitializing = false;
    return { detSession, recSession };
  })();

  return initPromise;
}

/**
 * Run end-to-end PaddleOCR on an image/canvas in the browser.
 * Extracts words/segments, then aggregates them into sentence-level bounding boxes.
 *
 * @param {HTMLImageElement|HTMLCanvasElement|ImageBitmap} imageSource
 * @param {object} [options]
 * @param {function} [options.onProgress] - Callback for progress updates
 * @returns {Promise<Array<{text: string, box: {x: number, y: number, width: number, height: number}, confidence: number}>>}
 */
export async function runPaddleOcr(imageSource, options = {}) {
  await initPaddleOcr();

  const onProgress = options.onProgress || (() => {});
  onProgress("Detecting text regions...");

  // Draw imageSource to standard canvas
  const origWidth = imageSource.naturalWidth || imageSource.videoWidth || imageSource.width;
  const origHeight = imageSource.naturalHeight || imageSource.videoHeight || imageSource.height;

  const canvas = document.createElement("canvas");
  canvas.width = origWidth;
  canvas.height = origHeight;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(imageSource, 0, 0, origWidth, origHeight);

  // 1. Run DBNet text detection
  const detectedBoxes = await detectTextRegions(canvas);

  if (detectedBoxes.length === 0) {
    onProgress("No text detected.");
    return [];
  }

  onProgress(`Recognizing text for ${detectedBoxes.length} region(s)...`);

  // 2. Run Thai text recognition on each detected box
  const rawSegments = [];
  for (let i = 0; i < detectedBoxes.length; i++) {
    const box = detectedBoxes[i];
    onProgress(`Recognizing line ${i + 1}/${detectedBoxes.length}...`);
    const result = await recognizeBox(ctx, box);
    if (result && result.text && result.text.trim()) {
      rawSegments.push({
        text: result.text.trim(),
        box: box,
        confidence: result.confidence,
      });
    }
  }

  onProgress("Grouping into sentences...");

  // 3. Union and assemble words into sentence-level bounding boxes
  const sentences = groupWordsIntoSentences(rawSegments, origWidth, origHeight, options);
  sentences.rawSegments = rawSegments;

  onProgress("Complete");
  return sentences;
}

/**
 * Run DBNet detection on canvas and return pixel bounding boxes in original image space.
 */
async function detectTextRegions(canvas) {
  const origW = canvas.width;
  const origH = canvas.height;

  // Resize image for DBNet: multiples of 32, max side ~960 for performance and accuracy
  const maxSide = 960;
  const scale = Math.min(maxSide / Math.max(origH, origW), 1.0);
  const targetH = Math.max(32, Math.round((origH * scale) / 32) * 32);
  const targetW = Math.max(32, Math.round((origW * scale) / 32) * 32);
  const ratioH = origH / targetH;
  const ratioW = origW / targetW;

  const resizedCanvas = document.createElement("canvas");
  resizedCanvas.width = targetW;
  resizedCanvas.height = targetH;
  const rCtx = resizedCanvas.getContext("2d", { willReadFrequently: true });
  rCtx.drawImage(canvas, 0, 0, targetW, targetH);

  const imgData = rCtx.getImageData(0, 0, targetW, targetH);
  const pixels = imgData.data;

  // Float32 array in NCHW format [1, 3, targetH, targetW] normalized with ImageNet stats
  const tensorSize = targetW * targetH;
  const floatData = new Float32Array(3 * tensorSize);

  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  for (let i = 0; i < tensorSize; i++) {
    const r = pixels[i * 4] / 255.0;
    const g = pixels[i * 4 + 1] / 255.0;
    const b = pixels[i * 4 + 2] / 255.0;

    floatData[0 * tensorSize + i] = (r - mean[0]) / std[0];
    floatData[1 * tensorSize + i] = (g - mean[1]) / std[1];
    floatData[2 * tensorSize + i] = (b - mean[2]) / std[2];
  }

  const inputTensor = new globalThis.ort.Tensor("float32", floatData, [
    1,
    3,
    targetH,
    targetW,
  ]);

  const output = await detSession.run({ x: inputTensor });
  const outName = detSession.outputNames[0];
  const predData = output[outName].data; // Float32Array of size targetH * targetW

  // Find connected components on thresholded probability map
  const rawBoxes = findBoxesFromBitmap(predData, targetW, targetH, 0.3);

  // Unclip / expand and scale back to original image space
  const boxes = [];
  for (const b of rawBoxes) {
    const expandX = Math.round(b.width * 0.12);
    const expandY = Math.round(b.height * 0.25);

    const bx = Math.max(0, Math.floor((b.x - expandX) * ratioW));
    const by = Math.max(0, Math.floor((b.y - expandY) * ratioH));
    const bw = Math.min(origW - bx, Math.ceil((b.width + expandX * 2) * ratioW));
    const bh = Math.min(origH - by, Math.ceil((b.height + expandY * 2) * ratioH));

    if (bw > 4 && bh > 4) {
      boxes.push({ x: bx, y: by, width: bw, height: bh });
    }
  }

  // Sort top-to-bottom, left-to-right
  boxes.sort((a, b) => (Math.abs(a.y - b.y) > 10 ? a.y - b.y : a.x - b.x));
  return boxes;
}

/**
 * Fast BFS connected-component bounding box finder on 1D Float32Array bitmap.
 */
function findBoxesFromBitmap(data, width, height, threshold = 0.3) {
  const total = width * height;
  const visited = new Uint8Array(total);
  const queue = new Int32Array(total);
  const boxes = [];

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      if (data[idx] >= threshold && !visited[idx]) {
        let minX = x;
        let maxX = x;
        let minY = y;
        let maxY = y;
        let count = 0;
        let qHead = 0;
        let qTail = 0;

        queue[qTail++] = idx;
        visited[idx] = 1;

        while (qHead < qTail) {
          const curr = queue[qHead++];
          const cy = Math.floor(curr / width);
          const cx = curr % width;
          count++;

          if (cx < minX) minX = cx;
          if (cx > maxX) maxX = cx;
          if (cy < minY) minY = cy;
          if (cy > maxY) maxY = cy;

          // 4-neighborhood
          const top = cy > 0 ? curr - width : -1;
          const bottom = cy < height - 1 ? curr + width : -1;
          const left = cx > 0 ? curr - 1 : -1;
          const right = cx < width - 1 ? curr + 1 : -1;

          if (top >= 0 && !visited[top] && data[top] >= threshold) {
            visited[top] = 1;
            queue[qTail++] = top;
          }
          if (bottom >= 0 && !visited[bottom] && data[bottom] >= threshold) {
            visited[bottom] = 1;
            queue[qTail++] = bottom;
          }
          if (left >= 0 && !visited[left] && data[left] >= threshold) {
            visited[left] = 1;
            queue[qTail++] = left;
          }
          if (right >= 0 && !visited[right] && data[right] >= threshold) {
            visited[right] = 1;
            queue[qTail++] = right;
          }
        }

        const bw = maxX - minX + 1;
        const bh = maxY - minY + 1;
        if (bw >= 3 && bh >= 3 && count >= 8) {
          boxes.push({ x: minX, y: minY, width: bw, height: bh });
        }
      }
    }
  }

  return boxes;
}

/**
 * Recognize Thai + Latin text within a single bounding box crop.
 */
async function recognizeBox(ctx, box) {
  const cropW = box.width;
  const cropH = box.height;
  if (cropW <= 0 || cropH <= 0) return null;

  // PaddleOCR RecResizeImg: height = 48, proportional width
  const targetH = 48;
  const targetW = Math.max(16, Math.round(targetH * (cropW / cropH)));

  const cropCanvas = document.createElement("canvas");
  cropCanvas.width = targetW;
  cropCanvas.height = targetH;
  const cCtx = cropCanvas.getContext("2d", { willReadFrequently: true });

  cCtx.drawImage(
    ctx.canvas,
    box.x,
    box.y,
    box.width,
    box.height,
    0,
    0,
    targetW,
    targetH
  );

  const imgData = cCtx.getImageData(0, 0, targetW, targetH);
  const pixels = imgData.data;

  // Normalize (pixel / 255.0 - 0.5) / 0.5 in NCHW [1, 3, 48, targetW]
  const tensorSize = targetW * targetH;
  const floatData = new Float32Array(3 * tensorSize);

  for (let i = 0; i < tensorSize; i++) {
    const r = (pixels[i * 4] / 255.0 - 0.5) / 0.5;
    const g = (pixels[i * 4 + 1] / 255.0 - 0.5) / 0.5;
    const b = (pixels[i * 4 + 2] / 255.0 - 0.5) / 0.5;

    floatData[0 * tensorSize + i] = r;
    floatData[1 * tensorSize + i] = g;
    floatData[2 * tensorSize + i] = b;
  }

  const recTensor = new globalThis.ort.Tensor("float32", floatData, [
    1,
    3,
    targetH,
    targetW,
  ]);

  const output = await recSession.run({ x: recTensor });
  const outName = recSession.outputNames[0];
  const logits = output[outName]; // shape [1, seqLen, numClasses = 526]
  const seqLen = logits.dims[1];
  const numClasses = logits.dims[2] || 526;
  const data = logits.data;

  // CTC Greedy Decode
  const decodedChars = [];
  const confidences = [];
  let lastToken = 0;

  for (let step = 0; step < seqLen; step++) {
    const offset = step * numClasses;
    let maxVal = -Infinity;
    let maxIdx = 0;

    for (let c = 0; c < numClasses; c++) {
      const val = data[offset + c];
      if (val > maxVal) {
        maxVal = val;
        maxIdx = c;
      }
    }

    // Sigmoid or softmax approximation for CTC score
    const conf = 1 / (1 + Math.exp(-Math.min(10, Math.max(-10, maxVal))));

    if (maxIdx !== 0 && maxIdx !== lastToken) {
      if (maxIdx < characterDict.length) {
        decodedChars.push(characterDict[maxIdx]);
        confidences.push(conf);
      }
    }
    lastToken = maxIdx;
  }

  const text = decodedChars.join("");
  const avgConf =
    confidences.length > 0
      ? confidences.reduce((a, b) => a + b, 0) / confidences.length
      : 0.0;

  return { text, confidence: avgConf };
}

/**
 * Union multiple bounding boxes into an enclosing bounding box (references geometry.py & ocrspace_backend.py).
 * @param {Array<{x: number, y: number, width: number, height: number}>} boxes
 * @returns {{x: number, y: number, width: number, height: number}|null}
 */
export function unionBoxes(boxes) {
  if (!Array.isArray(boxes) || boxes.length === 0) return null;

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  for (const b of boxes) {
    if (!b) continue;
    minX = Math.min(minX, b.x);
    minY = Math.min(minY, b.y);
    maxX = Math.max(maxX, b.x + b.width);
    maxY = Math.max(maxY, b.y + b.height);
  }

  if (minX === Infinity) return null;

  return {
    x: Math.round(minX),
    y: Math.round(minY),
    width: Math.round(maxX - minX),
    height: Math.round(maxY - minY),
  };
}

/**
 * Group word/token level segments into complete sentence-level segments with unified bounding boxes.
 * Words on the same line are merged into the same sentence ONLY if their horizontal gap is within
 * a natural word spacing threshold (maxGapRatio).
 *
 * Distant words (e.g. table columns, separate navigation items, unrelated UI widgets) are kept as
 * separate distinct sentences.
 *
 * @param {Array<{text: string, box: object, confidence: number}>} segments
 * @param {number} imageWidth
 * @param {number} imageHeight
 * @param {object} [options]
 * @param {number} [options.maxGapRatio=1.0] - Maximum gap between words as a multiple of line height (default: 1.0)
 * @returns {Array<{text: string, box: object, confidence: number, words: Array}>}
 */
export function groupWordsIntoSentences(segments, imageWidth, imageHeight, options = {}) {
  if (!Array.isArray(segments) || segments.length === 0) return [];

  const maxGapRatio = typeof options.maxGapRatio === "number" ? options.maxGapRatio : 1.0;

  // Step 1: Cluster into horizontal lines by vertical baseline / center alignment
  const sorted = [...segments].sort((a, b) => a.box.y - b.box.y || a.box.x - b.box.x);

  const lines = [];
  for (const seg of sorted) {
    let placed = false;
    for (const line of lines) {
      const ref = line[0];
      const avgH = (ref.box.height + seg.box.height) / 2;
      const centerYDiff = Math.abs(
        ref.box.y + ref.box.height / 2 - (seg.box.y + seg.box.height / 2)
      );
      if (centerYDiff <= avgH * 0.5) {
        line.push(seg);
        placed = true;
        break;
      }
    }
    if (!placed) {
      lines.push([seg]);
    }
  }

  // Step 2: Detect vertical column alignment guides across lines (e.g. multi-line tables, forms)
  // If multiple words across different lines share a similar starting X, that marks a column boundary!
  const colXStarts = [];
  for (const line of lines) {
    for (const seg of line) {
      colXStarts.push(seg.box.x);
    }
  }

  const columnMargins = [];
  for (let i = 0; i < colXStarts.length; i++) {
    const x = colXStarts[i];
    let matchCount = 0;
    for (const line of lines) {
      if (line.some((s) => Math.abs(s.box.x - x) <= 6)) {
        matchCount++;
      }
    }
    if (matchCount >= 2 && !columnMargins.some((cm) => Math.abs(cm - x) <= 6)) {
      columnMargins.push(x);
    }
  }

  // Step 3: For each horizontal line, sort left-to-right and split into separate columns/phrases
  const sentenceGroups = [];
  for (const line of lines) {
    line.sort((a, b) => a.box.x - b.box.x);
    let currentGroup = [line[0]];

    for (let i = 1; i < line.length; i++) {
      const prev = currentGroup[currentGroup.length - 1];
      const curr = line[i];
      const prevRight = prev.box.x + prev.box.width;
      const gap = curr.box.x - prevRight;
      const avgH = (prev.box.height + curr.box.height) / 2;
      const maxGap = Math.max(16, avgH * maxGapRatio);

      // Check if curr aligns with an established column boundary across lines
      const isColumnBoundary = columnMargins.some(
        (cm) => Math.abs(curr.box.x - cm) <= 6 && curr.box.x > currentGroup[0].box.x + 30
      );

      // Normal word space -> merge into same sentence
      // Wide gap or column margin alignment -> split into separate columns/phrases
      if (gap <= maxGap && !isColumnBoundary) {
        currentGroup.push(curr);
      } else {
        sentenceGroups.push(currentGroup);
        currentGroup = [curr];
      }
    }

    if (currentGroup.length > 0) {
      sentenceGroups.push(currentGroup);
    }
  }

  // Step 4: Union word boxes for each sentence group and build sentence-level segments
  const sentences = [];
  for (const group of sentenceGroups) {
    group.sort((a, b) => a.box.x - b.box.x);

    const unionBox = unionBoxes(group.map((w) => w.box));
    if (!unionBox) continue;

    const clampedBox = {
      x: Math.max(0, Math.min(imageWidth - 1, unionBox.x)),
      y: Math.max(0, Math.min(imageHeight - 1, unionBox.y)),
      width: Math.min(imageWidth - unionBox.x, unionBox.width),
      height: Math.min(imageHeight - unionBox.y, unionBox.height),
    };

    let sentenceText = "";
    for (let i = 0; i < group.length; i++) {
      const wText = group[i].text;
      if (i === 0) {
        sentenceText = wText;
      } else {
        sentenceText += " " + wText;
      }
    }

    // Segment group segments into individual word tokens with tight bounding boxes
    const detailedWords = [];
    for (const seg of group) {
      if (Array.isArray(seg.words) && seg.words.length > 0) {
        detailedWords.push(...seg.words);
      } else {
        const subWords = segmentIntoWords(seg.text, seg.box, seg.confidence);
        detailedWords.push(...subWords);
      }
    }

    const avgConfidence =
      group.reduce((sum, w) => sum + (w.confidence || 0), 0) / group.length;

    sentences.push({
      text: sentenceText.trim(),
      box: clampedBox,
      confidence: parseFloat(avgConfidence.toFixed(3)),
      words: detailedWords.length > 0 ? detailedWords : group,
    });
  }

  // Sort sentences in natural reading order: top-to-bottom, left-to-right
  sentences.sort((a, b) => {
    const diffY = a.box.y - b.box.y;
    const avgH = (a.box.height + b.box.height) / 2;
    if (Math.abs(diffY) > avgH * 0.5) {
      return diffY;
    }
    return a.box.x - b.box.x;
  });

  return sentences;
}

/**
 * Extract word-level bounding boxes for specific PII target strings within a sentence.
 * Allows pinpoint redaction so only the sensitive words are blacked out, preserving surrounding text.
 *
 * @param {{text: string, box: object, words?: Array<{text: string, box: object}>}} sentence
 * @param {string|string[]} piiTargets - One or more sensitive strings detected in this sentence
 * @returns {Array<{x: number, y: number, width: number, height: number}>} Bounding boxes to redact
 */
export function getRedactionBoxes(sentence, piiTargets) {
  if (!sentence) return [];
  const targets = Array.isArray(piiTargets) ? piiTargets : [piiTargets];
  const normalizedTargets = targets
    .filter((t) => typeof t === "string" && t.trim().length > 0)
    .map((t) => t.trim().toLowerCase());

  if (normalizedTargets.length === 0) return [];

  const words = Array.isArray(sentence.words) && sentence.words.length > 0
    ? sentence.words
    : null;

  // If word-level boxes are not available, fall back to the whole sentence box
  if (!words) {
    return sentence.box ? [sentence.box] : [];
  }

  const matchedBoxes = [];

  for (const w of words) {
    const wText = (w.text || "").trim().toLowerCase();
    if (!wText) continue;

    // Check if word is part of any PII target, or any PII target is part of this word
    const isPii = normalizedTargets.some(
      (target) => target.includes(wText) || wText.includes(target)
    );

    if (isPii && w.box) {
      matchedBoxes.push(w.box);
    }
  }

  // Safety fallback: if target was detected in sentence text but couldn't match individual words
  if (matchedBoxes.length === 0 && sentence.box) {
    const sentenceText = (sentence.text || "").toLowerCase();
    const hasPii = normalizedTargets.some((target) => sentenceText.includes(target));
    if (hasPii) {
      matchedBoxes.push(sentence.box);
    }
  }

  return matchedBoxes;
}

/**
 * Count visible horizontal glyphs in a string (ignoring Thai upper/lower combining vowel & tone marks).
 */
export function countHorizontalGlyphs(str) {
  if (!str) return 0;
  let count = 0;
  for (let i = 0; i < str.length; i++) {
    const code = str.charCodeAt(i);
    // Thai combining marks: 0x0E31, 0x0E34-0x0E3A, 0x0E47-0x0E4E
    const isThaiCombining =
      code === 0x0E31 ||
      (code >= 0x0E34 && code <= 0x0E3A) ||
      (code >= 0x0E47 && code <= 0x0E4E);
    if (!isThaiCombining) {
      count++;
    }
  }
  return Math.max(1, count);
}

/**
 * Segment a recognized text segment into individual word tokens with estimated bounding boxes.
 * Uses native Intl.Segmenter for intelligent Thai and multilingual word tokenization.
 *
 * @param {string} text
 * @param {{x: number, y: number, width: number, height: number}} box
 * @param {number} [confidence=1.0]
 * @returns {Array<{text: string, box: {x: number, y: number, width: number, height: number}, confidence: number}>}
 */
export function segmentIntoWords(text, box, confidence = 1.0) {
  if (!text || typeof text !== "string" || !box) return [];
  const trimmed = text.trim();
  if (!trimmed) return [];

  // Step 1: Tokenize using Intl.Segmenter (standard in all modern browsers)
  let rawTokens = [];
  if (typeof Intl !== "undefined" && Intl.Segmenter) {
    try {
      const segmenter = new Intl.Segmenter(["th", "en"], { granularity: "word" });
      for (const seg of segmenter.segment(text)) {
        const token = seg.segment.trim();
        if (token.length > 0 && seg.isWordLike) {
          rawTokens.push({
            text: token,
            startIndex: seg.index,
            length: seg.segment.length,
          });
        }
      }
    } catch (e) {
      rawTokens = [];
    }
  }

  // Fallback: tokenize by whitespace / punctuation
  if (rawTokens.length === 0) {
    const regex = /[^\s\u200B]+/g;
    let match;
    while ((match = regex.exec(text)) !== null) {
      rawTokens.push({
        text: match[0],
        startIndex: match.index,
        length: match[0].length,
      });
    }
  }

  // If only 1 token or empty, return the original box
  if (rawTokens.length <= 1) {
    return [{ text: trimmed, box: { ...box }, confidence }];
  }

  // Step 2: Compute proportional bounding boxes based on horizontal glyphs
  const totalGlyphs = countHorizontalGlyphs(text);
  const pxPerGlyph = box.width / totalGlyphs;

  const words = [];
  for (const token of rawTokens) {
    const prefix = text.substring(0, token.startIndex);
    const glyphsBefore = countHorizontalGlyphs(prefix);
    const tokenGlyphs = countHorizontalGlyphs(token.text);

    const wordX = Math.round(box.x + glyphsBefore * pxPerGlyph);
    const wordW = Math.max(8, Math.round(tokenGlyphs * pxPerGlyph));

    words.push({
      text: token.text,
      box: {
        x: Math.max(0, wordX),
        y: box.y,
        width: Math.min(box.x + box.width - wordX, wordW),
        height: box.height,
      },
      confidence: confidence,
    });
  }

  return words;
}

