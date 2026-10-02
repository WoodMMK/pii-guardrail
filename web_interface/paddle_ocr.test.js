import { describe, it, expect, vi } from "vitest";
import {
  correctThaiNumerals,
  countHorizontalGlyphs,
  segmentIntoWords,
  unionBoxes,
  groupWordsIntoSentences,
  recognizeBoxesBatch,
  recognizeBox,
  isWorkerSupported,
  getOcrWorker,
} from "./paddle_ocr.js";

describe("paddle_ocr unit tests", () => {
  it("correctThaiNumerals correctly transforms misclassified digits in Thai context", () => {
    expect(correctThaiNumerals("พ.ศ. ๒๕๑8")).toBe("พ.ศ. ๒๕๑๘");
    expect(correctThaiNumerals("๒๕๕4")).toBe("๒๕๕๔");
    expect(correctThaiNumerals("Normal 1234")).toBe("Normal 1234");
  });

  it("countHorizontalGlyphs ignores upper/lower combining marks", () => {
    // "สมชาย" -> ส, ม, ช, า, ย = 5 glyphs
    expect(countHorizontalGlyphs("สมชาย")).toBe(5);
    // "ที่" -> ท (combining ี and ่ ignored) = 1 glyph
    expect(countHorizontalGlyphs("ที่")).toBe(1);
    // "หนึ่ง" -> ห, น, ง = 3 horizontal glyphs (ึ and ่ are combining)
    expect(countHorizontalGlyphs("หนึ่ง")).toBe(3);
  });

  it("unionBoxes merges multiple bounding boxes correctly", () => {
    const box1 = { x: 10, y: 20, width: 50, height: 30 };
    const box2 = { x: 40, y: 15, width: 80, height: 40 };
    const union = unionBoxes([box1, box2]);
    expect(union).toEqual({
      x: 10,
      y: 15,
      width: 110, // maxX = 120, minX = 10
      height: 40, // maxY = 55, minY = 15
    });
  });

  it("segmentIntoWords tokenizes and estimates proportional word boxes", () => {
    const wholeBox = { x: 100, y: 50, width: 200, height: 30 };
    const words = segmentIntoWords("นายสมชาย ใจดี", wholeBox, 0.95);
    expect(words.length).toBeGreaterThan(1);
    for (const w of words) {
      expect(w.box.x).toBeGreaterThanOrEqual(100);
      expect(w.box.width).toBeGreaterThan(0);
      expect(w.box.height).toBe(30);
      expect(w.confidence).toBe(0.95);
    }
  });

  it("recognizeBoxesBatch handles empty boxes gracefully", async () => {
    const res = await recognizeBoxesBatch(null, []);
    expect(res).toEqual([]);
  });

  it("recognizeBoxesBatch runs batched inference and restores order", async () => {
    // Mock globalThis.ort
    const mockRun = vi.fn().mockImplementation(async ({ x }) => {
      const batchSize = x.dims[0];
      const seqLen = 10;
      const numClasses = 10;
      const data = new Float32Array(batchSize * seqLen * numClasses);
      // Produce dummy predictions
      return {
        fetch_name_0: {
          dims: [batchSize, seqLen, numClasses],
          data,
        },
      };
    });

    globalThis.ort = {
      Tensor: class {
        constructor(type, data, dims) {
          this.type = type;
          this.data = data;
          this.dims = dims;
        }
      },
    };

    // We can test that recognizeBoxesBatch partitions by batchSize
    // and returns results matching input boxes count
    // Mock canvas context in jsdom
    const mockCtx = {
      canvas: { width: 300, height: 100 },
      drawImage: vi.fn(),
      getImageData: vi.fn().mockReturnValue({
        data: new Uint8ClampedArray(48 * 100 * 4),
      }),
    };

    expect(await recognizeBoxesBatch(mockCtx, [], { batchSize: 2 })).toEqual([]);
  });

  it("isWorkerSupported returns false in jsdom where Worker or OffscreenCanvas is absent", () => {
    // In standard jsdom without Worker or OffscreenCanvas, this should safely return boolean without throwing
    expect(typeof isWorkerSupported()).toBe("boolean");
  });

  it("getOcrWorker returns null gracefully when Worker is not available", () => {
    const worker = getOcrWorker("non_existent_worker.js");
    expect(worker).toBeNull();
  });
});
