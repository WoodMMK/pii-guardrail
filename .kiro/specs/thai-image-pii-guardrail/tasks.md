# Implementation Plan: Thai Image PII Guardrail

## Overview

This plan builds the reusable Python core (`pii_guardrail`) first, fully decoupled from the web layer, then the FastAPI `Backend_Service` adapter, then the browser `Web_Interface`. Detection accuracy on scanned documents is the priority, so preprocessing (denoise + deskew) and the OCR abstraction are built and tested early. OCR runs server-side via an injectable `OCRBackend` (default PaddleOCR) so unit/property tests use a `FakeOCRBackend`.

Testing is dual-track: Hypothesis property-based tests (one test per each of the 19 correctness properties, min 100 iterations, tagged as required) for the core's deterministic logic, plus example/integration tests for model- and infrastructure-dependent behavior (PaddleOCR on Thai/Latin samples, the >=150 DPI scanned corpus, the 95% recall benchmark) and example-based frontend tests.

Tasks build incrementally; each ends wired into the pipeline so there is no orphaned code. Every task references the requirements and/or design properties it implements.

## Tasks

- [x] 1. Set up project structure, tooling, and core data models
  - Create the `pii_guardrail` package layout (`preprocessor`, `ocr`, `detector`, `classifier`, `redactor`, `pipeline`, `models`, `errors`) and a top-level `tests/` tree separating `unit/`, `property/`, and `integration/`.
  - Add dependency/config scaffolding: OpenCV + NumPy (imaging), Pillow (format load/save), Hypothesis and pytest (testing); PaddleOCR and FastAPI declared but isolated so the core imports without them.
  - Configure the test runner for single-execution runs (e.g., `pytest` / `pytest --no-header`), not watch mode.
  - _Requirements: reusable core boundary (design), 2.1_

  - [x] 1.1 Implement core data models and enums
    - Implement `SensitiveCategory` enum, `BoundingBox` (frozen, with invariants), `TextSegment`, `SensitiveRegion`, and `DetectionResult` (with `count` property) as framework-agnostic dataclasses.
    - _Requirements: 7.1, 7.2, 7.3_

  - [x] 1.2 Implement the core error hierarchy
    - Implement `GuardrailError`, `InvalidImageError`, `UnsupportedFormatError`, `FileTooLargeError`, `OCRProcessingError`, `ClassifierUnavailableError`.
    - _Requirements: 2.8, 2.9, 1.3, 1.5, 5.3_

  - [x]* 1.3 Write unit tests for data model invariants
    - Assert `BoundingBox` rejects non-positive width/height and negative origin; assert `DetectionResult.count` matches list length.
    - _Requirements: 7.1, 7.2, 7.3_

- [x] 2. Implement bounding-box normalization and geometry
  - [x] 2.1 Implement quadrilateral-to-axis-aligned `BoundingBox` normalization
    - Convert a PaddleOCR-style quadrilateral into the enclosing axis-aligned box, clamped inside image bounds (top-left origin).
    - _Requirements: 2.2_

  - [x]* 2.2 Write property test for bounding-box normalization
    - **Property 3: Bounding-box normalization stays within the image and encloses its source**
    - **Validates: Requirements 2.2**
    - Generate in-bounds quadrilaterals over images of width W / height H; assert x>=0, y>=0, width>0, height>0, x+width<=W, y+height<=H, and that the box contains every source vertex.

- [x] 3. Implement the Detector's pattern-based classification
  - [x] 3.1 Implement `classify_segment` pattern matchers
    - Implement deterministic pattern-based classification for phone number, URL, email address, money amount, percent value, API key, access token, and generic secret patterns; plus Thai/Latin person-name and organization-name heuristics/gazetteer. Return a `set[SensitiveCategory]`; empty/whitespace-only text returns the empty set; a segment may match multiple categories.
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10, 3.11, 4.1, 4.2, 4.3_

  - [x]* 3.2 Write property test: classification is total
    - **Property 7: Classification is total and returns a set of categories**
    - **Validates: Requirements 3.1**
    - For any string, `classify_segment` returns a set without raising.

  - [x]* 3.3 Write property test: category-bearing text is classified into its category
    - **Property 8: Category-bearing text is classified into its category**
    - **Validates: Requirements 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.1, 4.2, 4.3**
    - Use category-shaped strategies (phone/URL/email/money/percent/API key/access token) and gazetteer-seeded Thai/Latin name and org strategies; assert the returned set includes the generated category.

  - [x]* 3.4 Write property test: multi-category text yields all categories
    - **Property 9: Multi-category text yields all matching categories**
    - **Validates: Requirements 3.10**
    - Concatenate tokens from N distinct categories; assert all N categories are returned.

  - [x]* 3.5 Write property test: whitespace-only text is non-sensitive
    - **Property 10: Whitespace-only text is non-sensitive**
    - **Validates: Requirements 3.11**
    - Whitespace-only strategy (empty string, tabs, newlines, Unicode whitespace); assert empty set and no region created.

- [x] 4. Implement the optional Classifier_Model contract and Detector integration
  - [x] 4.1 Define `ClassifierModel` protocol and Detector wiring
    - Implement the `ClassifierModel` protocol (`available`, `classify`) and integrate it into `Detector.classify_segment`/`detect`: pattern-only when disabled; union with model results when enabled+available; on enabled-but-unavailable (or a raised `ClassifierUnavailableError`), fall back to patterns and append a warning to `DetectionOutcome.warnings`.
    - Implement `Detector.detect` to build `SensitiveRegion`s for sensitive segments (preserving boxes) and return a `DetectionOutcome`.
    - _Requirements: 5.1, 5.2, 5.3, 3.1, 3.10, 3.11_

  - [x]* 4.2 Write property test: disabled classifier equals pattern-only
    - **Property 11: Disabled classifier equals pattern-only classification**
    - **Validates: Requirements 5.2**
    - For any string, disabled-classifier result equals pattern-based result.

  - [x]* 4.3 Write property test: enabled-but-unavailable falls back and warns
    - **Property 12: Enabled-but-unavailable classifier falls back and warns**
    - **Validates: Requirements 5.3**
    - Use a `FakeClassifierModel` with `available=False`; assert result equals pattern categories and `warnings` is non-empty.

  - [x]* 4.4 Write unit test: enabled+available classifier is consulted
    - Use a spy `FakeClassifierModel` with `available=True`; assert `classify` is called and its categories are merged.
    - _Requirements: 5.1_

- [x] 5. Checkpoint - core classification stable
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement the OCR_Engine with injectable backend
  - [x] 6.1 Implement `OCRBackend` protocol and `OCREngine.extract`
    - Implement the `OCRBackend` protocol (`run`, `available`) and `OCREngine.extract`: return `[]` when no text (not an error), normalize each segment's box via task 2.1, carry confidence in [0.0, 1.0]; raise `InvalidImageError` for missing/corrupt/unsupported input and `OCRProcessingError` on failure after accepting valid input (no partial result).
    - Provide test doubles: `FakeOCRBackend` (controlled segment lists incl. empty) and `RaisingOCRBackend`.
    - _Requirements: 2.1, 2.2, 2.6, 2.7, 2.8, 2.9_

  - [x]* 6.2 Write property test: no text yields zero regions
    - **Property 4: No recognized text yields zero sensitive regions**
    - **Validates: Requirements 2.6**
    - With a `FakeOCRBackend` returning `[]`, assert resulting `DetectionResult.count == 0`.

  - [x]* 6.3 Write property test: confidence within range
    - **Property 5: Confidence values are within range**
    - **Validates: Requirements 2.7**
    - For generated segments/derived regions, assert 0.0 <= confidence <= 1.0.

  - [x]* 6.4 Write property test: invalid input fails closed
    - **Property 6: Invalid input fails closed with no partial result**
    - **Validates: Requirements 2.8, 2.9**
    - Missing/corrupt/unsupported input and `RaisingOCRBackend` both raise and produce no `DetectionResult`.

- [x] 7. Implement the Preprocessor (denoise, deskew, quality estimation)
  - [x] 7.1 Implement `Preprocessor.preprocess` and `estimate_skew_angle`
    - Implement noise reduction before OCR, skew detection + rotation so residual skew <= 1 degree, DPI estimation, and `quality_sufficient` flag (False below the 150 DPI threshold). Return `PreprocessResult`; raise `InvalidImageError` on empty/corrupt arrays.
    - _Requirements: 6.2, 6.3, 6.6_

  - [x]* 7.2 Write property test: skew correction within 1 degree
    - **Property 13: Skew correction aligns text rows within 1 degree**
    - **Validates: Requirements 6.3**
    - Rotation strategy applying known skew angles to synthetic text-row images; assert `|residual_skew_deg| <= 1.0`.

  - [x]* 7.3 Write unit test: denoise runs before OCR
    - Use a spy to assert the denoise step is applied before extraction; optional metamorphic noise-metric check.
    - _Requirements: 6.2_

- [x] 8. Implement the Redactor
  - [x] 8.1 Implement `Redactor.redact`
    - Draw a filled black rectangle covering the full `BoundingBox` of each region on a copy of the image; zero regions returns an equivalent image; preserve width, height, and channel count.
    - _Requirements: 8.1, 8.2, 8.4, 8.5_

  - [x]* 8.2 Write property test: redaction covers each box in black
    - **Property 17: Redaction covers each region's full bounding box in black**
    - **Validates: Requirements 8.1, 8.2**
    - Image + non-empty region strategies; assert every pixel within each box (corners inclusive) is black.

  - [x]* 8.3 Write property test: zero regions is identity
    - **Property 18: Redaction with zero regions is an identity**
    - **Validates: Requirements 8.4**
    - Assert the redacted image is pixelwise-equivalent to the input.

  - [x]* 8.4 Write property test: redaction preserves dimensions
    - **Property 19: Redaction preserves image dimensions**
    - **Validates: Requirements 8.5**
    - Assert output width, height, and channel count equal the input.

  - [x]* 8.5 Write unit test: redactor returns an encodable image
    - Assert the output can be encoded to PNG/JPEG.
    - _Requirements: 8.3_

- [x] 9. Checkpoint - all core components stable
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. Implement DetectionResult serialization and completeness guarantees
  - [x] 10.1 Implement `DetectionResult` serialize/deserialize
    - Implement mapping to/from the machine-readable structured format: `BoundingBox` as `{x, y, width, height}`, `categories` as a list of category strings, including `image_width`/`image_height` and `count`.
    - _Requirements: 7.4_

  - [x]* 10.2 Write property test: every region is well-formed and complete
    - **Property 15: Every region in a Detection_Result is well-formed and complete**
    - **Validates: Requirements 7.1, 7.2, 7.3**
    - Assert `regions` matches detection output, `count` equals list length, every box satisfies invariants, and every region has a non-empty category set.

  - [x]* 10.3 Write property test: serialization round-trips
    - **Property 16: Detection_Result serialization round-trips**
    - **Validates: Requirements 7.4**
    - Serialize then deserialize any `DetectionResult`; assert an equivalent result (regions, boxes, categories, dimensions).

- [x] 11. Implement the GuardrailPipeline core entry point
  - [x] 11.1 Implement `GuardrailPipeline.process`
    - Wire preprocess -> OCR -> detect -> (optional) redact; when `quality_sufficient` is False, short-circuit to a `PipelineResult` with `quality_sufficient=False`, zero regions, and an insufficient-quality warning; propagate `InvalidImageError`/`OCRProcessingError`; carry detector warnings through. Return `PipelineResult`.
    - _Requirements: 2.6, 6.4, 6.6, 7.1, 8.1, 8.4_

  - [x]* 11.2 Write property test: insufficient quality yields no reliable regions + indication
    - **Property 14: Insufficient quality yields no reliable regions and a quality indication**
    - **Validates: Requirements 6.6**
    - For below-threshold/unprocessable images, assert `quality_sufficient == False`, zero regions, and a non-empty insufficient-quality warning.

  - [x]* 11.3 Write integration test: end-to-end core flow with fakes
    - Use `FakeOCRBackend` + `FakeClassifierModel` to drive `process` and assert `DetectionResult` + `Redacted_Image` are produced coherently.
    - _Requirements: 2.6, 7.1, 8.1_

- [x] 12. Implement the Backend_Service (FastAPI adapter)
  - [x] 12.1 Implement upload validation (format + size)
    - Validate PNG/JPEG format and max upload size at the transport edge before the core runs; load valid bytes into an image array; map failures to `UnsupportedFormatError` / `FileTooLargeError`.
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [x]* 12.2 Write property test: unsupported formats rejected with format identified
    - **Property 1: Unsupported formats are rejected with the format identified**
    - **Validates: Requirements 1.3, 1.4**
    - Non-PNG/JPEG uploads are rejected and the error identifies the received format.

  - [x]* 12.3 Write property test: oversized uploads rejected with limit identified
    - **Property 2: Oversized uploads are rejected with the limit identified**
    - **Validates: Requirements 1.5**
    - Over-limit uploads rejected with the size limit in the error; within-limit uploads pass the size check.

  - [x] 12.4 Implement `POST /api/redact` endpoint
    - Parse multipart upload, invoke `GuardrailPipeline.process`, serialize `DetectionResult` to JSON, base64-encode the `Redacted_Image`, include `quality_sufficient` and `warnings`; return equivalent (re-encoded) image with zero regions when `count == 0`.
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 8.3, 9.1_

  - [x] 12.5 Implement error mapping and `/api/health`
    - Map core exceptions to HTTP: 415 `UNSUPPORTED_FORMAT`, 413 `FILE_TOO_LARGE`, 400 `INVALID_IMAGE`, 422 `OCR_FAILED`, 500 `INTERNAL_ERROR`; implement `GET /api/health` reporting `ocr_backend_available`.
    - _Requirements: 1.3, 1.5, 2.8, 2.9, 9.6_

  - [x]* 12.6 Write unit test: API response contains detection_result and redacted_image
    - Assert a successful response includes both fields and the warnings array.
    - _Requirements: 9.1_

- [x] 13. Implement the default PaddleOCR backend
  - [x] 13.1 Implement `PaddleOCRBackend`
    - Wrap a PaddleOCR-family model (Thai + Latin recognition, detection quadrilaterals, angle classification); expose `available` reflecting model load; wire as the `OCREngine` default when available.
    - _Requirements: 2.3, 2.4, 2.5_

- [x] 14. Checkpoint - backend and OCR backend wired
  - Ensure all tests pass, ask the user if questions arise.

- [x] 15. Integration tests over sample corpora (model/infrastructure-dependent)
  - [x]* 15.1 Write PaddleOCR extraction tests on Thai and Latin samples
    - Assert text + boxes extracted on representative Thai and Latin sample images; assert default backend is a PaddleOCR backend when available and `/api/health` reports availability.
    - _Requirements: 2.1, 2.3, 2.4, 2.5, 6.1_

  - [x]* 15.2 Write scanned-corpus detection test (>=150 DPI)
    - On a labeled >=150 DPI scanned corpus, assert the pipeline identifies present `Sensitive_Regions`.
    - _Requirements: 6.4_

  - [x]* 15.3 Write scanned-vs-synthetic recall benchmark
    - Over a paired corpus, compute detection recall on >=150 DPI scans relative to synthetic counterparts; assert >= 95%.
    - _Requirements: 6.5_

- [x] 16. Implement the Web_Interface (browser frontend)
  - [x] 16.1 Implement upload form, status indicator, and request flow
    - Build the single-page frontend: file selection/submit posting multipart to `POST /api/redact`, and a processing status indicator shown while the request is in flight.
    - _Requirements: 1.1, 9.5_

  - [x] 16.2 Implement results rendering and download control
    - Render the redacted image, display the count of `Sensitive_Regions` and their `Sensitive_Category` values, provide a download control for the redacted image, and render error messages / warnings from the response.
    - _Requirements: 9.2, 9.3, 9.4, 9.6_

  - [x]* 16.3 Write frontend tests
      - Assert: renders redacted image (9.2), renders region count + category summary (9.3), download control works (9.4), status indicator shown during in-flight request (9.5), error message rendered for error responses (9.6).
    - _Requirements: 9.2, 9.3, 9.4, 9.5, 9.6_

- [x] 17. Final checkpoint - full stack wired
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional (test-related) and can be skipped for a faster MVP; core implementation tasks are never optional.
- The reusable core (tasks 1-11) is built and tested before any web dependency is introduced (tasks 12+), preserving the framework-agnostic boundary.
- OCR runs through an injectable `OCRBackend`; unit/property tests use `FakeOCRBackend`/`RaisingOCRBackend`, and the real `PaddleOCRBackend` is added at task 13.
- Each of Properties 1-19 is implemented as exactly one Hypothesis property test (min 100 iterations) tagged `# Feature: thai-image-pii-guardrail, Property {number}: {property_text}`.
- PaddleOCR quality, Thai/Latin recognition, and the 95% recall target are validated by integration tests/benchmark (task 15), not property tests, since they exercise external model behavior.
- Image persistence/storage is out of scope; processing is transient per request.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["1.3", "2.1", "3.1", "7.1", "8.1"] },
    { "id": 2, "tasks": ["2.2", "3.2", "3.3", "3.4", "3.5", "4.1", "7.2", "7.3", "8.2", "8.3", "8.4", "8.5", "10.1"] },
    { "id": 3, "tasks": ["4.2", "4.3", "4.4", "6.1", "10.2", "10.3"] },
    { "id": 4, "tasks": ["6.2", "6.3", "6.4", "11.1", "13.1"] },
    { "id": 5, "tasks": ["11.2", "11.3", "12.1", "12.4", "12.5"] },
    { "id": 6, "tasks": ["12.2", "12.3", "12.6", "15.1", "15.2", "15.3", "16.1", "16.2"] },
    { "id": 7, "tasks": ["16.3"] }
  ]
}
```
