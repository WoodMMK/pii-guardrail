# Requirements Document

## Introduction

The Thai Image PII Guardrail is a web application that detects sensitive data inside images and redacts it by drawing filled black rectangles over the detected regions. The application runs as a browser-based Web Interface backed by a Python backend service: a user opens a web page, uploads an image, and receives the detection results and the redacted image back in the browser. PaddleOCR and the rest of the detection and redaction pipeline run server-side in the backend service. The primary language focus is Thai, and the application must handle both synthetic images (text rendered or pasted directly onto an image) and scanned document images, which are noisy, skewed, lower quality, and use varying fonts.

The application locates sensitive content across three broad groups: personal information (person names, organization names, phone numbers, URLs, email addresses, money amounts, and percent values), credentials/security tokens (API keys, access tokens, and similar secrets), and Thailand-specific identifiers (Thai national ID numbers, dates of birth, Thai postal addresses, bank account numbers, vehicle license plates, and passport numbers). For each detected item the application produces a bounding box, then covers each region with a solid black box and returns a redacted output image to the browser.

The underlying detection and redaction engine is a reusable Python core that the backend service invokes; the same core logic can be reused independently of the web layer. The immediate priority is detection accuracy, specifically reliably locating sensitive data on scanned documents where the current approach underperforms. The preferred OCR technology is the PaddleOCR family of models running server-side, with optional assistance from a small language model for classification of detected text. Redaction (black-box overlay) is a required capability but is secondary to detection in the current priority.

Image storage and retention policy (whether and how long uploaded images or redacted outputs are persisted) is out of scope for the current version and to be determined later.

## Glossary

- **Guardrail**: The overall system that detects and redacts sensitive data in images, comprising the Web_Interface, the Backend_Service, and the reusable Python detection/redaction core they invoke.
- **Web_Interface**: The browser-based frontend that lets a user upload an Input_Image and view the Detection_Result and Redacted_Image.
- **Backend_Service**: The Python server-side component that receives an uploaded Input_Image from the Web_Interface, runs the OCR_Engine, Detector, and Redactor, and returns the Detection_Result and Redacted_Image.
- **OCR_Engine**: The component that extracts text and per-text-region bounding boxes from an input image. The preferred implementation belongs to the PaddleOCR family of models.
- **Detector**: The component that classifies extracted text segments into sensitive data categories and selects which text regions are sensitive.
- **Classifier_Model**: An optional small language model used by the Detector to assist in classifying extracted text into sensitive data categories.
- **Redactor**: The component that draws filled black rectangles over sensitive regions to produce the redacted image.
- **Input_Image**: An image file supplied to the Guardrail for processing. It may be a synthetic image (text rendered or pasted directly) or a scanned document image.
- **Scanned_Document_Image**: An Input_Image produced by scanning or photographing a physical document, characterized by noise, skew, variable lighting, lower resolution, and varying fonts.
- **Synthetic_Image**: An Input_Image where text is rendered or pasted directly onto the image with clean, machine-generated glyphs.
- **Text_Region**: A rectangular area of an Input_Image, expressed as a bounding box, that contains a segment of recognized text.
- **Bounding_Box**: A set of pixel coordinates describing a rectangular region of an Input_Image, used to locate a Text_Region and to place a redaction rectangle.
- **Sensitive_Region**: A Text_Region that the Detector has classified as containing sensitive data belonging to one of the supported categories.
- **Sensitive_Category**: One of the supported classes of sensitive data: person name, organization name, phone number, URL, email address, money amount, percent value, API key, access token, or similar secret; plus the Thailand-specific identifiers Thai national ID number, date of birth, Thai postal address, bank account number, vehicle license plate, and passport number.
- **Redacted_Image**: The output image produced by the Guardrail with every Sensitive_Region covered by a filled black rectangle.
- **Detection_Result**: The structured set of Sensitive_Regions with their Sensitive_Category and Bounding_Box for a given Input_Image.

## Requirements

### Requirement 1: Upload and Accept Input Images via the Web Interface

**User Story:** As a user, I want to upload an image through a web page, so that the tool can process it for sensitive data without installing anything.

#### Acceptance Criteria

1. WHEN a user selects an image file in the Web_Interface and submits it, THE Web_Interface SHALL send the file to the Backend_Service as an Input_Image.
2. WHEN the Backend_Service receives an uploaded file, THE Backend_Service SHALL load the file as an Input_Image.
3. IF the uploaded file is not a supported image format, THEN THE Backend_Service SHALL reject the upload and return an error indication identifying the unsupported format to the Web_Interface.
4. THE Backend_Service SHALL support the PNG and JPEG image formats as Input_Image formats.
5. IF the uploaded file exceeds the maximum accepted upload size, THEN THE Backend_Service SHALL reject the upload and return an error indication identifying the size limit to the Web_Interface.

### Requirement 2: Extract Text and Locations with OCR

**User Story:** As a user, I want the tool to read text and its position from the image, so that sensitive data can be located precisely for redaction.

#### Acceptance Criteria

1. WHEN an Input_Image is loaded, THE OCR_Engine SHALL extract recognized text segments from the Input_Image, producing for each segment a Detection_Result entry containing the recognized text string.
2. WHEN the OCR_Engine extracts a text segment, THE OCR_Engine SHALL produce a Bounding_Box that locates the corresponding Text_Region, expressed as pixel coordinates referenced to the top-left origin (0,0) of the Input_Image.
3. THE OCR_Engine SHALL recognize Thai-language text in the Input_Image.
4. THE OCR_Engine SHALL recognize Latin-script text in the Input_Image.
5. WHERE a PaddleOCR-family model is available, THE OCR_Engine SHALL use a PaddleOCR-family model to perform text extraction.
6. IF the OCR_Engine extracts no text segments from the Input_Image, THEN THE Guardrail SHALL produce a Detection_Result containing zero Sensitive_Regions.
7. WHEN the OCR_Engine recognizes a text segment, THE OCR_Engine SHALL assign the segment a confidence value between 0.0 and 1.0 inclusive in the Detection_Result entry.
8. IF the Input_Image is missing, corrupt, or of an unsupported format, THEN THE OCR_Engine SHALL reject the Input_Image, produce no Detection_Result, and return an error indication identifying the invalid input.
9. IF the OCR_Engine fails during text extraction after accepting a valid Input_Image, THEN THE OCR_Engine SHALL halt processing of that Input_Image, produce no partial Detection_Result, and return an error indication identifying the processing failure.

### Requirement 3: Detect Personal Information

**User Story:** As a user, I want the tool to detect personal information in the image, so that private data can be redacted.

#### Acceptance Criteria

1. WHEN the Detector processes an extracted text segment, THE Detector SHALL classify the text segment into one or more Sensitive_Categories or classify it as non-sensitive.
2. IF a text segment contains a person name, THEN THE Detector SHALL classify the corresponding Text_Region as a Sensitive_Region with the person name category.
3. IF a text segment contains an organization name, THEN THE Detector SHALL classify the corresponding Text_Region as a Sensitive_Region with the organization name category.
4. IF a text segment contains a phone number, THEN THE Detector SHALL classify the corresponding Text_Region as a Sensitive_Region with the phone number category.
5. IF a text segment contains a URL, THEN THE Detector SHALL classify the corresponding Text_Region as a Sensitive_Region with the URL category.
6. IF a text segment contains an email address, THEN THE Detector SHALL classify the corresponding Text_Region as a Sensitive_Region with the email address category.
7. IF a text segment contains a money amount, THEN THE Detector SHALL classify the corresponding Text_Region as a Sensitive_Region with the money amount category.
8. IF a text segment contains a percent value, THEN THE Detector SHALL classify the corresponding Text_Region as a Sensitive_Region with the percent value category.
9. IF a text segment contains a person name or an organization name written in Thai script, THEN THE Detector SHALL classify the corresponding Text_Region as a Sensitive_Region with the person name category or the organization name category respectively.
10. IF a text segment matches more than one Sensitive_Category, THEN THE Detector SHALL classify the corresponding Text_Region as a Sensitive_Region and assign every matching Sensitive_Category to that Text_Region.
11. IF a text segment is empty or contains only whitespace characters, THEN THE Detector SHALL classify the text segment as non-sensitive and SHALL NOT create a Sensitive_Region for the corresponding Text_Region.
12. IF a text segment contains a 13-digit Thai national ID number whose mod-11 check digit is valid, THEN THE Detector SHALL classify the corresponding Text_Region as a Sensitive_Region with the Thai national ID category, and SHALL NOT additionally classify it as a phone number. A 13-digit number whose check digit is invalid SHALL NOT be classified as a Thai national ID.
13. IF a text segment contains a date of birth (a numeric date, or a date using a Thai or Latin month name), THEN THE Detector SHALL classify the corresponding Text_Region as a Sensitive_Region with the date of birth category.
14. IF a text segment contains a Thai postal address (indicated by an address keyword such as บ้านเลขที่/ถนน/ตำบล/อำเภอ/จังหวัด, or an abbreviated marker ต./อ./จ. that is not part of a Thai month abbreviation), THEN THE Detector SHALL classify the corresponding Text_Region as a Sensitive_Region with the Thai address category.
15. IF a text segment contains a bank account number (a grouped numeric account of the common Thai form), THEN THE Detector SHALL classify the corresponding Text_Region as a Sensitive_Region with the bank account category.
16. IF a text segment contains a Thai vehicle license plate (Thai consonants adjacent to digits, with an optional leading digit), THEN THE Detector SHALL classify the corresponding Text_Region as a Sensitive_Region with the license plate category.
17. IF a text segment contains a passport number (one or two uppercase letters followed by six or seven digits), THEN THE Detector SHALL classify the corresponding Text_Region as a Sensitive_Region with the passport number category.
18. The Thailand-specific categories in criteria 12-17 are detected by deterministic pattern rules (with checksum validation for the national ID). They participate in multi-category assignment (criterion 10) like any other Sensitive_Category, and a whitespace-only segment (criterion 11) remains non-sensitive.

### Requirement 4: Detect Credentials and Security Tokens

**User Story:** As a user, I want the tool to detect credentials and security tokens in the image, so that secrets are not exposed.

#### Acceptance Criteria

1. WHERE a text segment contains an API key, THE Detector SHALL classify the corresponding Text_Region as a Sensitive_Region with the API key category.
2. WHERE a text segment contains an access token, THE Detector SHALL classify the corresponding Text_Region as a Sensitive_Region with the access token category.
3. WHERE a text segment matches a known secret pattern other than an API key or access token, THE Detector SHALL classify the corresponding Text_Region as a Sensitive_Region with the secret category.

### Requirement 5: Support Optional Classifier Model

**User Story:** As a user, I want the tool to optionally use a small language model to help classification, so that detection of ambiguous Thai text improves.

#### Acceptance Criteria

1. WHERE the Classifier_Model is enabled, THE Detector SHALL submit extracted text segments to the Classifier_Model for classification into a Sensitive_Category.
2. WHERE the Classifier_Model is disabled, THE Detector SHALL classify extracted text segments using pattern-based classification.
3. IF the Classifier_Model is enabled but unavailable at runtime, THEN THE Detector SHALL fall back to pattern-based classification and record a warning message.

### Requirement 6: Detect Sensitive Data on Scanned Documents

**User Story:** As a user, I want the tool to detect sensitive data on scanned document images, so that noisy and skewed documents are handled as reliably as clean images.

#### Acceptance Criteria

1. WHEN a Scanned_Document_Image is loaded, THE OCR_Engine SHALL extract recognized text segments from the Scanned_Document_Image.
2. WHILE processing a Scanned_Document_Image, THE Guardrail SHALL apply image preprocessing that reduces noise before text extraction.
3. WHILE processing a Scanned_Document_Image, THE Guardrail SHALL apply skew correction that aligns text rows to within 1 degree of the horizontal axis before text extraction.
4. WHEN processing a Scanned_Document_Image containing text at a resolution of 150 dots per inch or higher, THE Guardrail SHALL produce a Detection_Result identifying the Sensitive_Regions present in the Scanned_Document_Image.
5. THE Guardrail SHALL produce a Detection_Result for a Scanned_Document_Image at a resolution of 150 dots per inch or higher that identifies at least 95 percent of the Sensitive_Regions identified in the corresponding Synthetic_Image processed through the same detection pipeline.
6. IF a Scanned_Document_Image has a resolution below 150 dots per inch, THEN THE Guardrail SHALL first attempt to upscale the image toward the 150 DPI threshold (bounded by a maximum upscale factor) before detection; IF the image still cannot reach the threshold after the bounded upscale, or cannot be processed by the OCR_Engine, THEN THE Guardrail SHALL produce a Detection_Result indicating that no Sensitive_Regions could be reliably identified and provide an indication that the image quality is insufficient for detection.

   - Auto-upscale behavior (Option A): WHEN a below-threshold image is upscaled to reach the 150 DPI threshold, THE Guardrail SHALL run detection on the upscaled image and return the upscaled image as the basis for the Redacted_Image. As a consequence, the returned/redacted image has LARGER pixel dimensions than the uploaded image, and a warning SHALL indicate that the image was upscaled and that detection accuracy may be reduced (upscaling does not recover detail the source lacked).

### Requirement 7: Produce Detection Results with Bounding Boxes

**User Story:** As a user, I want the tool to output the located sensitive regions with their bounding boxes, so that I can verify detection before redaction and drive the redaction step.

#### Acceptance Criteria

1. WHEN detection completes for an Input_Image, THE Backend_Service SHALL produce a Detection_Result that lists each Sensitive_Region.
2. WHEN the Backend_Service produces a Detection_Result, THE Backend_Service SHALL include the Bounding_Box for each Sensitive_Region.
3. WHEN the Backend_Service produces a Detection_Result, THE Backend_Service SHALL include the Sensitive_Category for each Sensitive_Region.
4. WHEN the Backend_Service produces a Detection_Result, THE Backend_Service SHALL represent the Detection_Result in a machine-readable structured format that can be returned to the Web_Interface.

### Requirement 8: Redact Sensitive Regions

**User Story:** As a user, I want the tool to cover the detected sensitive regions with black boxes, so that the sensitive content is no longer visible in the output image.

#### Acceptance Criteria

1. WHEN a Detection_Result contains at least one Sensitive_Region, THE Redactor SHALL draw a filled black rectangle over each Sensitive_Region in the Input_Image.
2. WHEN the Redactor draws a filled black rectangle over a Sensitive_Region, THE Redactor SHALL cover the full area of the Bounding_Box of the Sensitive_Region.
3. WHEN redaction completes, THE Redactor SHALL produce a Redacted_Image that the Backend_Service can return to the Web_Interface.
4. WHEN a Detection_Result contains zero Sensitive_Regions, THE Redactor SHALL produce a Redacted_Image equivalent to the Input_Image.
5. THE Redactor SHALL preserve the pixel dimensions of the Input_Image in the Redacted_Image.

### Requirement 9: Deliver Results Through the Web Interface

**User Story:** As a user, I want to see the detection results and the redacted image in my browser, so that I can review and download them immediately.

#### Acceptance Criteria

1. WHEN the Backend_Service completes processing of an uploaded Input_Image, THE Backend_Service SHALL return the Detection_Result and the Redacted_Image to the Web_Interface.
2. WHEN the Web_Interface receives the Redacted_Image, THE Web_Interface SHALL display the Redacted_Image to the user.
3. WHEN the Web_Interface receives the Detection_Result, THE Web_Interface SHALL display the count of Sensitive_Regions and their Sensitive_Category values to the user.
4. WHEN the Web_Interface displays a Redacted_Image, THE Web_Interface SHALL provide a control that lets the user download the Redacted_Image.
5. WHILE the Backend_Service is processing an uploaded Input_Image, THE Web_Interface SHALL display a processing status indicator to the user.
6. IF the Backend_Service returns an error indication for an uploaded Input_Image, THEN THE Web_Interface SHALL display an error message describing the failure to the user.
