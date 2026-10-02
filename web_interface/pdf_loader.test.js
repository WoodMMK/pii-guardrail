import { describe, it, expect, vi } from "vitest";
import { isPdfFile, convertPdfToPageCanvases, loadPdfDocument, initPdfJs } from "./pdf_loader.js";

describe("pdf_loader unit tests", () => {
  it("isPdfFile correctly identifies PDF files and extensions", () => {
    expect(isPdfFile("document.pdf")).toBe(true);
    expect(isPdfFile("DOCUMENT.PDF")).toBe(true);
    expect(isPdfFile("image.png")).toBe(false);
    expect(isPdfFile({ name: "scan.pdf", type: "application/pdf" })).toBe(true);
    expect(isPdfFile({ name: "scan.png", type: "image/png" })).toBe(false);
    expect(isPdfFile(null)).toBe(false);
    expect(isPdfFile(undefined)).toBe(false);
  });

  it("convertPdfToPageCanvases throws when PDF.js is not available", async () => {
    const originalPdfjs = globalThis.pdfjsLib;
    delete globalThis.pdfjsLib;

    await expect(
      convertPdfToPageCanvases(new ArrayBuffer(10))
    ).rejects.toThrow("PDF.js is not loaded");

    globalThis.pdfjsLib = originalPdfjs;
  });

  it("convertPdfToPageCanvases renders pages with mock PDF.js", async () => {
    const mockPage = {
      getViewport: vi.fn().mockReturnValue({ width: 400, height: 600 }),
      render: vi.fn().mockReturnValue({
        promise: Promise.resolve(),
      }),
    };

    const mockPdfDoc = {
      numPages: 2,
      getPage: vi.fn().mockResolvedValue(mockPage),
    };

    globalThis.pdfjsLib = {
      GlobalWorkerOptions: {},
      getDocument: vi.fn().mockReturnValue({
        promise: Promise.resolve(mockPdfDoc),
      }),
    };

    const origCreateElement = document.createElement.bind(document);
    vi.spyOn(document, "createElement").mockImplementation((tag) => {
      if (tag === "canvas") {
        return {
          width: 0,
          height: 0,
          getContext: vi.fn().mockReturnValue({}),
        };
      }
      return origCreateElement(tag);
    });

    const pages = await convertPdfToPageCanvases(new ArrayBuffer(100), {
      scale: 2.0,
    });

    vi.restoreAllMocks();

    expect(pages.length).toBe(2);
    expect(pages[0].pageNum).toBe(1);
    expect(pages[0].width).toBe(400);
    expect(pages[0].height).toBe(600);
    expect(pages[1].pageNum).toBe(2);
  });

  it("loadPdfDocument provides on-demand per-page rendering", async () => {
    const mockPage = {
      getViewport: vi.fn().mockReturnValue({ width: 500, height: 750 }),
      render: vi.fn().mockReturnValue({
        promise: Promise.resolve(),
      }),
    };

    const mockPdfDoc = {
      numPages: 3,
      getPage: vi.fn().mockResolvedValue(mockPage),
    };

    globalThis.pdfjsLib = {
      GlobalWorkerOptions: {},
      getDocument: vi.fn().mockReturnValue({
        promise: Promise.resolve(mockPdfDoc),
      }),
    };

    const origCreateElement = document.createElement.bind(document);
    vi.spyOn(document, "createElement").mockImplementation((tag) => {
      if (tag === "canvas") {
        return {
          width: 0,
          height: 0,
          getContext: vi.fn().mockReturnValue({}),
        };
      }
      return origCreateElement(tag);
    });

    const doc = await loadPdfDocument(new ArrayBuffer(50));
    expect(doc.numPages).toBe(3);

    const p1 = await doc.renderPage(1, 2.0);
    expect(p1.pageNum).toBe(1);
    expect(p1.width).toBe(500);
    expect(p1.height).toBe(750);
    expect(mockPdfDoc.getPage).toHaveBeenCalledWith(1);

    vi.restoreAllMocks();
  });
});

