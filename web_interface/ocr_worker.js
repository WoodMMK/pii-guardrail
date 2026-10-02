/**
 * Web Worker for 100% Client-Side In-Browser PaddleOCR Execution.
 * Runs ONNX Runtime Web in a dedicated OS background thread (0% main thread blocking).
 */

/* global ort */
try {
  importScripts("vendor/ort.min.js");
} catch (e) {
  // May be running in an environment with different root
  try {
    importScripts("/vendor/ort.min.js");
  } catch (err) {
    console.error("Failed to importScripts vendor/ort.min.js:", err);
  }
}

const ARABIC_TO_THAI = {
  "0": "๐", "1": "๑", "2": "๒", "3": "๓", "4": "๔",
  "5": "๕", "6": "๖", "7": "๗", "8": "๘", "9": "๙",
};

function correctThaiNumerals(text) {
  if (!text || typeof text !== "string") return text;
  const toThai = (digits) =>
    digits.split("").map((c) => ARABIC_TO_THAI[c] || c).join("");

  text = text.replace(/(พ\.ศ\.\s*)([๐-๙0-9]{2,4})/g, (match, prefix, digits) => {
    if (/[๐-๙]/.test(digits)) return prefix + toThai(digits);
    return match;
  });

  text = text.replace(/[๐-๙0-9]{2,}/g, (match) => {
    if (/[๐-๙]/.test(match) && /[0-9]/.test(match)) {
      return toThai(match);
    }
    return match;
  });
  return text;
}

function correctThaiOcrText(text) {
  if (!text || typeof text !== "string") return text;
  text = correctThaiNumerals(text);

  // Fix generic Buddhist Year glitches where 7 is misread as bracket or pipe e.g. 256] -> 2567
  text = text.replace(/(25\d{2})[\]|Il!]/g, "$1");
  text = text.replace(/(256)[\]|Il!]/g, "$17");
  text = text.replace(/([0-3]?\d[\/.:][0-1]?\d[\/.:])(256)[\]|Il!]/g, "$1$27");

  return text;
}

let detSession = null;
let recSession = null;
let characterDict = null;
let isInitializing = false;
let initPromise = null;
let activeTaskId = null;
let cancelledTasks = new Set();

async function initModels(options = {}) {
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

    if (self.ort && self.ort.env && self.ort.env.wasm) {
      const normalizedVendor = vendorPath.endsWith("/") ? vendorPath : `${vendorPath}/`;
      self.ort.env.wasm.wasmPaths = {
        mjs: `${normalizedVendor}ort-wasm-simd-threaded.mjs`,
        wasm: `${normalizedVendor}ort-wasm-simd-threaded.wasm`,
      };
      const supportsThreading = typeof crossOriginIsolated !== "undefined" && crossOriginIsolated;
      self.ort.env.wasm.numThreads = supportsThreading
        ? Math.min(4, navigator.hardwareConcurrency || 2)
        : 1;
    }

    if (!self.ort) {
      throw new Error("ONNX Runtime Web (ort) is not loaded in Web Worker.");
    }

    // 1. Fetch character dictionary
    const dictResp = await fetch(`${modelsPath}languages/thai/dict.txt`);
    if (!dictResp.ok) {
      throw new Error(`Failed to load Thai dictionary: ${dictResp.statusText}`);
    }
    const dictText = await dictResp.text();
    const rawLines = dictText.split(/\r?\n/).filter((l) => l.length > 0);
    characterDict = ["blank", ...rawLines, " "];

    // 2. Create Detection & Recognition inference sessions
    let ep = "wasm";
    try {
      const gpuOptions = {
        executionProviders: ["webgpu"],
        graphOptimizationLevel: "all",
      };
      [detSession, recSession] = await Promise.all([
        self.ort.InferenceSession.create(`${modelsPath}detection/v3/det.onnx`, gpuOptions),
        self.ort.InferenceSession.create(`${modelsPath}languages/thai/rec.onnx`, gpuOptions),
      ]);
      ep = "webgpu";
    } catch {
      const wasmOptions = {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      };
      [detSession, recSession] = await Promise.all([
        self.ort.InferenceSession.create(`${modelsPath}detection/v3/det.onnx`, wasmOptions),
        self.ort.InferenceSession.create(`${modelsPath}languages/thai/rec.onnx`, wasmOptions),
      ]);
      ep = "wasm";
    }

    isInitializing = false;
    return { detSession, recSession, executionProvider: ep };
  })();

  return initPromise;
}

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

async function detectTextRegions(canvas) {
  const origW = canvas.width;
  const origH = canvas.height;

  const maxSide = 960;
  const scale = Math.min(maxSide / Math.max(origH, origW), 1.0);
  const targetH = Math.max(32, Math.round((origH * scale) / 32) * 32);
  const targetW = Math.max(32, Math.round((origW * scale) / 32) * 32);
  const ratioH = origH / targetH;
  const ratioW = origW / targetW;

  const resizedCanvas = new OffscreenCanvas(targetW, targetH);
  const rCtx = resizedCanvas.getContext("2d", { willReadFrequently: true });
  rCtx.drawImage(canvas, 0, 0, targetW, targetH);

  const imgData = rCtx.getImageData(0, 0, targetW, targetH);
  const pixels = imgData.data;

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

  const inputTensor = new self.ort.Tensor("float32", floatData, [1, 3, targetH, targetW]);
  const output = await detSession.run({ x: inputTensor });
  const outName = detSession.outputNames[0];
  const predData = output[outName].data;

  const rawBoxes = findBoxesFromBitmap(predData, targetW, targetH, 0.3);

  const boxes = [];
  for (const b of rawBoxes) {
    const expandX = Math.max(6, Math.round(Math.max(b.width * 0.04, b.height * 0.25)));
    const expandYTop = Math.max(4, Math.round(b.height * 0.25));
    const expandYBottom = Math.max(4, Math.round(b.height * 0.22));

    const bx = Math.max(0, Math.floor((b.x - expandX) * ratioW));
    const by = Math.max(0, Math.floor((b.y - expandYTop) * ratioH));
    const bw = Math.min(origW - bx, Math.ceil((b.width + expandX * 2) * ratioW));
    const bh = Math.min(origH - by, Math.ceil((b.height + expandYTop + expandYBottom) * ratioH));

    if (bw > 4 && bh > 4) {
      boxes.push({ x: bx, y: by, width: bw, height: bh });
    }
  }

  boxes.sort((a, b) => (Math.abs(a.y - b.y) > 10 ? a.y - b.y : a.x - b.x));
  return boxes;
}

let scratchCanvas = null;
let scratchCtx = null;

function getScratchCanvas(width, height) {
  if (!scratchCanvas) {
    scratchCanvas = new OffscreenCanvas(width, height);
    scratchCtx = scratchCanvas.getContext("2d", { willReadFrequently: true });
  }
  scratchCanvas.width = width;
  scratchCanvas.height = height;
  return { canvas: scratchCanvas, ctx: scratchCtx };
}

async function recognizeBox(sourceCanvas, box) {
  if (!box || box.width <= 0 || box.height <= 0) return null;

  const cropW = box.width;
  const cropH = box.height;

  const targetH = 48;
  const targetW = Math.max(16, Math.round(targetH * (cropW / cropH)));

  const { ctx: sCtx } = getScratchCanvas(targetW, targetH);
  sCtx.drawImage(sourceCanvas, box.x, box.y, box.width, box.height, 0, 0, targetW, targetH);

  const imgData = sCtx.getImageData(0, 0, targetW, targetH);
  const pixels = imgData.data;

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

  const recTensor = new self.ort.Tensor("float32", floatData, [1, 3, targetH, targetW]);
  const output = await recSession.run({ x: recTensor });
  const outName = recSession.outputNames[0];
  const logits = output[outName];
  const seqLen = logits.dims[1];
  const numClasses = logits.dims[2] || 526;
  const data = logits.data;

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

  return { text, confidence: avgConf, box };
}

function countHorizontalGlyphs(str) {
  if (!str) return 0;
  return str.replace(/[\u0E31\u0E34-\u0E3A\u0E47-\u0E4E]/g, "").length || 1;
}

function segmentIntoWords(text, box, confidence) {
  if (!text || !box) return [];
  const trimmed = text.trim();
  if (!trimmed) return [];

  let rawTokens = [];
  if (typeof Intl !== "undefined" && typeof Intl.Segmenter === "function") {
    try {
      const segmenter = new Intl.Segmenter("th", { granularity: "word" });
      const segments = Array.from(segmenter.segment(text));
      for (const seg of segments) {
        if (seg.isWordLike || seg.segment.trim().length > 0) {
          rawTokens.push({
            text: seg.segment,
            startIndex: seg.index,
            length: seg.segment.length,
          });
        }
      }
    } catch {
      rawTokens = [];
    }
  }

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

  if (rawTokens.length <= 1) {
    return [{ text: trimmed, box: { ...box }, confidence }];
  }

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

function unionBoxes(boxes) {
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

function groupWordsIntoSentences(segments, imageWidth, imageHeight, options = {}) {
  if (!Array.isArray(segments) || segments.length === 0) return [];
  const maxGapRatio = typeof options.maxGapRatio === "number" ? options.maxGapRatio : 1.0;

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
    if (matchCount >= 3 && !columnMargins.some((m) => Math.abs(m - x) <= 10)) {
      columnMargins.push(x);
    }
  }

  const sentences = [];

  for (const line of lines) {
    line.sort((a, b) => a.box.x - b.box.x);

    let currentGroup = [line[0]];

    for (let i = 1; i < line.length; i++) {
      const prev = currentGroup[currentGroup.length - 1];
      const curr = line[i];

      const gap = curr.box.x - (prev.box.x + prev.box.width);
      const avgHeight = (prev.box.height + curr.box.height) / 2;
      const isColumnBoundary = columnMargins.some(
        (margin) => curr.box.x >= margin - 4 && curr.box.x <= margin + 12 && gap > 12
      );

      const maxGapAllowed = avgHeight * maxGapRatio;

      if (gap <= maxGapAllowed && !isColumnBoundary) {
        currentGroup.push(curr);
      } else {
        const sentenceBox = unionBoxes(currentGroup.map((s) => s.box));
        const sentenceText = currentGroup.map((s) => s.text).join(" ");
        const avgConf =
          currentGroup.reduce((sum, s) => sum + (s.confidence || 0), 0) / currentGroup.length;

        const allWords = [];
        for (const seg of currentGroup) {
          allWords.push(...segmentIntoWords(seg.text, seg.box, seg.confidence));
        }

        sentences.push({
          text: sentenceText,
          box: sentenceBox,
          confidence: avgConf,
          words: allWords,
        });

        currentGroup = [curr];
      }
    }

    if (currentGroup.length > 0) {
      const sentenceBox = unionBoxes(currentGroup.map((s) => s.box));
      const sentenceText = currentGroup.map((s) => s.text).join(" ");
      const avgConf =
        currentGroup.reduce((sum, s) => sum + (s.confidence || 0), 0) / currentGroup.length;

      const allWords = [];
      for (const seg of currentGroup) {
        allWords.push(...segmentIntoWords(seg.text, seg.box, seg.confidence));
      }

      sentences.push({
        text: sentenceText,
        box: sentenceBox,
        confidence: avgConf,
        words: allWords,
      });
    }
  }

  return sentences;
}

// Web Worker message listener
self.onmessage = async (event) => {
  const { type, id, options = {} } = event.data;

  if (type === "init") {
    try {
      const res = await initModels(options);
      self.postMessage({ type: "init_done", executionProvider: res.executionProvider });
    } catch (err) {
      self.postMessage({ type: "error", error: err.message });
    }
    return;
  }

  if (type === "cancel") {
    if (id) cancelledTasks.add(id);
    return;
  }

  if (type === "ocr") {
    const { width, height, buffer, bitmap } = event.data;
    activeTaskId = id;

    const reportProgress = (msg) => {
      self.postMessage({ type: "progress", id, message: msg });
    };

    try {
      if (cancelledTasks.has(id)) {
        throw new Error("OCR operation was cancelled.");
      }

      await initModels(options);

      reportProgress("Detecting text regions...");

      // Prepare canvas in worker
      const canvas = new OffscreenCanvas(width, height);
      const ctx = canvas.getContext("2d", { willReadFrequently: true });

      if (bitmap) {
        ctx.drawImage(bitmap, 0, 0);
        if (typeof bitmap.close === "function") bitmap.close();
      } else if (buffer) {
        const imgData = new ImageData(new Uint8ClampedArray(buffer), width, height);
        ctx.putImageData(imgData, 0, 0);
      }

      if (cancelledTasks.has(id)) {
        throw new Error("OCR operation was cancelled.");
      }

      // 1. Text detection
      const detectedBoxes = await detectTextRegions(canvas);

      if (cancelledTasks.has(id)) {
        throw new Error("OCR operation was cancelled.");
      }

      if (detectedBoxes.length === 0) {
        reportProgress("No text detected.");
        self.postMessage({ type: "ocr_result", id, sentences: [], rawSegments: [] });
        return;
      }

      reportProgress(`Recognizing text for ${detectedBoxes.length} region(s)...`);

      // 2. Sequential recognition
      const rawSegments = [];
      for (let i = 0; i < detectedBoxes.length; i++) {
        if (cancelledTasks.has(id)) {
          throw new Error("OCR operation was cancelled.");
        }

        const box = detectedBoxes[i];
        if (i % 3 === 0 || i === detectedBoxes.length - 1) {
          reportProgress(`Recognizing line ${i + 1}/${detectedBoxes.length}...`);
        }

        const result = await recognizeBox(canvas, box);
        if (result && result.text && result.text.trim()) {
          const recognized = correctThaiOcrText(result.text.trim());
          rawSegments.push({
            text: recognized,
            box: box,
            confidence: result.confidence,
          });
        }
      }

      reportProgress("Grouping into sentences...");

      // 3. Sentence grouping
      const sentences = groupWordsIntoSentences(rawSegments, width, height, options);
      sentences.rawSegments = rawSegments;

      reportProgress("Complete");

      self.postMessage({
        type: "ocr_result",
        id,
        sentences,
        rawSegments,
      });
    } catch (err) {
      self.postMessage({
        type: "error",
        id,
        error: err.message || "Failed to execute OCR in Web Worker",
      });
    } finally {
      cancelledTasks.delete(id);
    }
  }
};
