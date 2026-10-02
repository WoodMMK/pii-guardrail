/**
 * Client-Side PDF to Canvas Converter using PDF.js.
 *
 * Converts multi-page PDF documents into high-resolution HTML Canvas elements
 * directly in the browser (100% local, zero external network requests).
 */

/**
 * Check if a file object or filename represents a PDF document.
 * @param {File|Blob|string} file
 * @returns {boolean}
 */
export function isPdfFile(file) {
  if (!file) return false;
  if (typeof file === "string") {
    return file.toLowerCase().endsWith(".pdf");
  }
  const type = file.type || "";
  const name = file.name || "";
  return type === "application/pdf" || name.toLowerCase().endsWith(".pdf");
}

/**
 * Configure PDF.js worker script path.
 * Uses local vendor/pdf.worker.min.js for offline privacy.
 */
export function initPdfJs() {
  if (globalThis.pdfjsLib && !globalThis.pdfjsLib.GlobalWorkerOptions?.workerSrc) {
    globalThis.pdfjsLib.GlobalWorkerOptions = globalThis.pdfjsLib.GlobalWorkerOptions || {};
    globalThis.pdfjsLib.GlobalWorkerOptions.workerSrc = "vendor/pdf.worker.min.js";
  }
}

/**
 * Load a PDF document for progressive on-demand page rendering.
 *
 * @param {File|Blob|ArrayBuffer} file
 * @returns {Promise<{numPages: number, renderPage: (pageNum: number, scale?: number) => Promise<{pageNum: number, canvas: HTMLCanvasElement, width: number, height: number}>}>}
 */
export async function loadPdfDocument(file) {
  initPdfJs();

  if (!globalThis.pdfjsLib) {
    throw new Error(
      "PDF.js is not loaded. Please ensure vendor/pdf.min.js is included in your HTML."
    );
  }

  let data;
  if (file instanceof ArrayBuffer) {
    data = file;
  } else if (file && typeof file.arrayBuffer === "function") {
    data = await file.arrayBuffer();
  } else {
    throw new Error("Invalid PDF input: expected a File, Blob, or ArrayBuffer.");
  }

  const loadingTask = globalThis.pdfjsLib.getDocument({ data });
  const pdfDoc = await loadingTask.promise;
  const numPages = pdfDoc.numPages || 0;

  if (numPages < 1) {
    throw new Error("The uploaded PDF document contains no pages.");
  }

  return {
    numPages,
    async renderPage(pageNum, scale = 2.0) {
      const page = await pdfDoc.getPage(pageNum);
      const viewport = page.getViewport({ scale });

      const canvas = document.createElement("canvas");
      canvas.width = Math.round(viewport.width);
      canvas.height = Math.round(viewport.height);
      const ctx = canvas.getContext("2d", { willReadFrequently: true });

      await page.render({
        canvasContext: ctx,
        viewport: viewport,
      }).promise;

      return {
        pageNum,
        canvas,
        width: canvas.width,
        height: canvas.height,
      };
    },
  };
}

/**
 * Convert a PDF file/blob/ArrayBuffer into an array of page canvas objects.
 *
 * @param {File|Blob|ArrayBuffer} file - PDF source
 * @param {object} [options]
 * @param {number} [options.scale=2.0] - Render scale factor (2.0 = ~150-200 DPI, optimal for OCR)
 * @param {function} [options.onProgress] - Progress callback
 * @returns {Promise<Array<{pageNum: number, canvas: HTMLCanvasElement, width: number, height: number}>>}
 */
export async function convertPdfToPageCanvases(file, options = {}) {
  const scale = typeof options.scale === "number" ? options.scale : 2.0;
  const onProgress = options.onProgress || (() => {});

  onProgress("Loading PDF document...");
  const doc = await loadPdfDocument(file);
  const numPages = doc.numPages;

  const pages = [];
  for (let p = 1; p <= numPages; p++) {
    onProgress(`Converting PDF page ${p} of ${numPages} to image...`);
    const pageData = await doc.renderPage(p, scale);
    pages.push(pageData);
  }

  onProgress(`Converted ${numPages} PDF page(s) to images successfully.`);
  return pages;
}
