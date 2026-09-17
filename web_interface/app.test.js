// Frontend tests for the Web_Interface (task 16.3).
//
// Validates: Requirements 9.2, 9.3, 9.4, 9.5, 9.6.
//
// These run under Vitest with the jsdom environment (see vitest.config.js) so
// renderResult/renderError/handleSubmit can be exercised against a real DOM and
// fetch can be mocked via injected impls.

import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  getElements,
  renderResult,
  renderError,
  renderOcrDebug,
  handleSubmit,
  submitImage,
  summarizeCategories,
} from "./app.js";

// Minimal DOM fragment carrying the ids the module's DOM contract depends on.
const DOM_FRAGMENT = `
  <form id="upload-form">
    <input id="file-input" type="file" />
    <button id="submit-btn" type="submit">Redact image</button>
  </form>
  <div id="status-indicator" role="status" aria-busy="false" hidden>
    <span class="status-text">Processing...</span>
  </div>
  <div id="result-container"></div>
  <div id="error-container" role="alert"></div>
  <details id="ocr-debug" hidden>
    <summary>OCR debug <span id="ocr-debug-count"></span></summary>
    <div id="ocr-debug-body"></div>
  </details>
`;

beforeEach(() => {
  document.body.innerHTML = DOM_FRAGMENT;
});

describe("renderResult (Requirements 9.2, 9.3, 9.4)", () => {
  it("renders the redacted image with the correct data URL (9.2)", () => {
    const elements = getElements(document);
    const response = {
      redacted_image: { format: "png", base64: "aGVsbG8=" },
      detection_result: { count: 0, regions: [] },
    };

    renderResult(response, elements);

    const img = document.getElementById("redacted-image");
    expect(img).not.toBeNull();
    expect(img.tagName).toBe("IMG");
    expect(elements.resultContainer.contains(img)).toBe(true);
    expect(img.getAttribute("src")).toBe("data:image/png;base64,aGVsbG8=");
  });

  it("renders the region count and category summary (9.3)", () => {
    const elements = getElements(document);
    const response = {
      redacted_image: { format: "png", base64: "aGVsbG8=" },
      detection_result: {
        count: 2,
        regions: [
          { categories: ["email", "url"] },
          { categories: ["phone_number"] },
        ],
      },
    };

    renderResult(response, elements);

    const countEl = document.getElementById("region-count");
    expect(countEl).not.toBeNull();
    expect(countEl.textContent).toContain("2");

    const summaryEl = document.getElementById("category-summary");
    expect(summaryEl).not.toBeNull();
    const summaryText = summaryEl.textContent;
    expect(summaryText).toContain("email");
    expect(summaryText).toContain("url");
    expect(summaryText).toContain("phone_number");
  });

  it("summarizeCategories aggregates category counts across regions (9.3)", () => {
    const summary = summarizeCategories([
      { categories: ["email", "url"] },
      { categories: ["phone_number"] },
      { categories: ["email"] },
    ]);
    const byName = Object.fromEntries(summary.map((s) => [s.category, s.count]));
    expect(byName.email).toBe(2);
    expect(byName.url).toBe(1);
    expect(byName.phone_number).toBe(1);
  });

  it("provides a download control with download attribute + data URL (9.4)", () => {
    const elements = getElements(document);
    const response = {
      redacted_image: { format: "png", base64: "aGVsbG8=" },
      detection_result: { count: 0, regions: [] },
    };

    renderResult(response, elements);

    const link = document.getElementById("download-link");
    expect(link).not.toBeNull();
    expect(link.tagName).toBe("A");
    expect(link.hasAttribute("download")).toBe(true);
    expect(link.getAttribute("download")).toBe("redacted.png");
    expect(link.getAttribute("href")).toBe("data:image/png;base64,aGVsbG8=");
  });
});

describe("handleSubmit status indicator (Requirement 9.5)", () => {
  it("shows the status indicator while in flight and hides it after settle", async () => {
    const elements = getElements(document);
    const statusEl = elements.statusIndicator;

    // Deferred promise we control so we can inspect the in-flight state.
    let resolveSubmit;
    const pending = new Promise((resolve) => {
      resolveSubmit = resolve;
    });
    const submit = vi.fn(() => pending);

    // Kick off the submission but do NOT await yet.
    const submitPromise = handleSubmit({ preventDefault() {} }, elements, {
      submit,
      onResult: () => {},
    });

    // In flight: indicator visible + busy, submit button disabled.
    expect(statusEl.hidden).toBe(false);
    expect(statusEl.getAttribute("aria-busy")).toBe("true");
    expect(elements.submitBtn.disabled).toBe(true);

    // Resolve the request and let handleSubmit finish.
    resolveSubmit({ redacted_image: null, detection_result: { count: 0, regions: [] } });
    await submitPromise;

    // Settled: indicator hidden again, button re-enabled.
    expect(statusEl.hidden).toBe(true);
    expect(statusEl.getAttribute("aria-busy")).toBe("false");
    expect(elements.submitBtn.disabled).toBe(false);
    expect(submit).toHaveBeenCalledTimes(1);
  });

  it("hides the status indicator even when the request fails (9.5/9.6)", async () => {
    const elements = getElements(document);
    const statusEl = elements.statusIndicator;

    const submit = vi.fn(() => Promise.reject(new Error("boom")));
    const onError = vi.fn();

    await handleSubmit({ preventDefault() {} }, elements, { submit, onError });

    expect(statusEl.hidden).toBe(true);
    expect(statusEl.getAttribute("aria-busy")).toBe("false");
    expect(elements.submitBtn.disabled).toBe(false);
    expect(onError).toHaveBeenCalledTimes(1);
  });
});

describe("renderError (Requirement 9.6)", () => {
  it("renders the error message plus structured detail and code", () => {
    const elements = getElements(document);
    const error = new Error("Bad input");
    error.payload = {
      error: { code: "INVALID_IMAGE", detail: "received: GIF" },
    };

    renderError(error, elements);

    const container = elements.errorContainer;
    expect(container.textContent).toContain("Bad input");

    const detailEl = document.getElementById("error-detail");
    expect(detailEl).not.toBeNull();
    expect(detailEl.textContent).toContain("received: GIF");

    const codeEl = document.getElementById("error-code");
    expect(codeEl).not.toBeNull();
    expect(codeEl.textContent).toContain("INVALID_IMAGE");
  });

  it("falls back to a generic message when none is provided", () => {
    const elements = getElements(document);
    renderError(new Error(""), elements);
    const messageEl = document.getElementById("error-message");
    expect(messageEl).not.toBeNull();
    expect(messageEl.textContent.length).toBeGreaterThan(0);
  });
});

describe("submitImage fetch handling (Requirements 9.2, 9.6)", () => {
  const fakeFile = { name: "photo.png" };

  it("resolves with the parsed body for a successful response", async () => {
    const body = {
      redacted_image: { format: "png", base64: "aGVsbG8=" },
      detection_result: { count: 1, regions: [{ categories: ["email"] }] },
    };
    const fetchImpl = vi.fn(async () => ({
      ok: true,
      status: 200,
      json: async () => body,
    }));

    const result = await submitImage(fakeFile, { apiUrl: "/api/redact", fetchImpl });

    expect(result).toEqual(body);
    expect(fetchImpl).toHaveBeenCalledTimes(1);
    const [url, options] = fetchImpl.mock.calls[0];
    expect(url).toBe("/api/redact");
    expect(options.method).toBe("POST");
  });

  it("rejects with an Error carrying status and payload on error responses (9.6)", async () => {
    const errorBody = {
      error: { code: "INVALID_IMAGE", message: "Bad input", detail: "received: GIF" },
    };
    const fetchImpl = vi.fn(async () => ({
      ok: false,
      status: 400,
      json: async () => errorBody,
    }));

    await expect(
      submitImage(fakeFile, { apiUrl: "/api/redact", fetchImpl })
    ).rejects.toMatchObject({
      message: "Bad input",
      status: 400,
      payload: errorBody,
    });
  });
});

describe("renderOcrDebug (raw OCR text debug view)", () => {
  it("renders each OCR segment's text, confidence, and box, and reveals the panel", () => {
    const elements = getElements(document);
    const segments = [
      { text: "john@example.com", confidence: 0.98, box: { x: 1, y: 1, width: 50, height: 10 }, categories: ["email"], redacted: true },
      { text: "just some words", confidence: 0.55, box: { x: 1, y: 20, width: 50, height: 10 }, categories: [], redacted: false },
    ];

    renderOcrDebug(segments, elements);

    expect(elements.ocrDebug.hidden).toBe(false);
    expect(elements.ocrDebugCount.textContent).toContain("2 segments");

    const rows = elements.ocrDebugBody.querySelectorAll("tbody tr");
    expect(rows.length).toBe(2);

    // Non-sensitive text is shown too (the whole point of the debug view).
    const bodyText = elements.ocrDebugBody.textContent;
    expect(bodyText).toContain("john@example.com");
    expect(bodyText).toContain("just some words");
    expect(bodyText).toContain("98.0%");
    expect(bodyText).toContain("1, 1, 50, 10");

    // Per-segment classification is shown: category tag + redacted flag.
    expect(bodyText).toContain("email");
    expect(bodyText).toContain("(not sensitive)");
    const redactedCells = elements.ocrDebugBody.querySelectorAll(".ocr-redacted-yes");
    const notRedactedCells = elements.ocrDebugBody.querySelectorAll(".ocr-redacted-no");
    expect(redactedCells.length).toBe(1);
    expect(notRedactedCells.length).toBe(1);
    expect(redactedCells[0].textContent).toBe("yes");
    expect(notRedactedCells[0].textContent).toBe("no");
  });

  it("shows an empty note when OCR recognized no text", () => {
    const elements = getElements(document);
    renderOcrDebug([], elements);

    expect(elements.ocrDebug.hidden).toBe(false);
    expect(elements.ocrDebugCount.textContent).toContain("0 segments");
    expect(elements.ocrDebugBody.textContent).toContain("did not recognize any text");
  });

  it("escapes segment text as textContent (no markup injection)", () => {
    const elements = getElements(document);
    renderOcrDebug(
      [{ text: "<img src=x onerror=alert(1)>", confidence: 0.5, box: { x: 0, y: 0, width: 1, height: 1 } }],
      elements
    );

    // The malicious string is present as text, not as an actual <img> element.
    expect(elements.ocrDebugBody.querySelector("img")).toBeNull();
    expect(elements.ocrDebugBody.textContent).toContain("<img src=x onerror=alert(1)>");
  });
});
