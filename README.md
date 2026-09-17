# PII Guardrail สำหรับรูปภาพ (POC)

ระบบ Guardrail สำหรับสแกนรูปภาพเอกสารเพื่อ **ตรวจจับและเบลอ (redact) ข้อมูลส่วนบุคคล (PII)** ก่อนส่งต่อรูปภาพไปยัง AI ภายนอก รองรับทั้งสถาปัตยกรรมแบบ **100% In-Browser OCR (WebAssembly / Zero External API)** และ Backend Pipeline ผสมผสาน Multi-layer Classifiers

```
[ เอกสาร / รูปภาพ ]
        │
        ▼
[ In-Browser PaddleOCR (ONNX Runtime Web) ]
   ├── ตรวจจับขอบเขตข้อความ & แยกคอลัมน์อัตโนมัติ (Column Margins & Gap Analysis)
   ├── สับคำภาษาไทยรายคำ (Intl.Segmenter + Proportional Glyph Boxes)
   └── สร้างผลลัพธ์โครงสร้างลำดับชั้น (Sentences + Word-Level Bounding Boxes)
        │
        ▼
[ Classifier Layers (Pattern + LLM + Presidio + detect-secrets) ]
        │
        ▼
[ Pinpoint Redactor (ถมดำเฉพาะคำที่เป็น PII โดยไม่ปิดทับ Label รอบข้าง) ]
```

---

## สารบัญ

1. [ความต้องการของระบบ (Requirements)](#requirements)
2. [การติดตั้งและ Setup](#setup)
3. [วิธีรันและทดลองใช้งานหน้าเว็บ (Web Interface)](#การรันและทดลองใช้งานหน้าเว็บ-web-interface)
4. [โครงสร้างผลลัพธ์ OCR (Data Structure)](#โครงสร้างผลลัพธ์-ocr-data-structure)
5. [คำแนะนำสำหรับทีมที่นำไปทำ Pattern Filter ต่อ (Word-Level Pinpoint Redaction)](#คำแนะนำสำหรับทีมที่นำไปทำ-pattern-filter-ต่อ-word-level-pinpoint-redaction)
6. [สถาปัตยกรรมการตรวจจับ (Detection Layers)](#สถาปัตยกรรมการตรวจจับ-detection-layers)
7. [การทดสอบ (Tests)](#tests)
8. [ความปลอดภัยและ Privacy](#ความปลอดภัยและ-privacy)

---

## Requirements

- **Python 3.10+**
- **Modern Web Browser** (Chrome, Edge, Firefox, Safari) รองรับ WebAssembly สำหรับ In-Browser OCR
- **Docker** (สำหรับ LiteLLM proxy — จำเป็นเฉพาะเมื่อต้องการเปิดใช้ LLM Layer)
- API keys (ทางเลือก - ไม่จำเป็นหากใช้ In-Browser OCR):
  - `OCRSPACE_API_KEY` — ฟรีที่ https://ocr.space/ocrapi (สำหรับ Cloud OCR fallback)
  - `LITELLM_API_KEY` — จาก LiteLLM proxy ของคุณ (สำหรับ LLM classifier)

---

## Setup

```powershell
# 1. Clone repository และเข้าสู่โฟลเดอร์
git clone https://github.com/WoodMMK/pii-guardrail.git
cd pii-guardrail

# 2. สร้าง Python Virtual Environment
python -m venv .venv

# 3. ติดตั้ง Dependencies ทั้งหมด (dev รวม ocr, web, presidio, secrets, pytest)
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

# 4. ตั้งค่า Environment (คีย์จริง — ไฟล์ .env จะถูก gitignore ไม่ขึ้น git)
Copy-Item .env.example .env
# หากต้องการใช้ Cloud OCR หรือ LLM ให้เปิดแก้ .env ใส่คีย์ที่เกี่ยวข้อง
```

> **หมายเหตุ:** โมเดล PaddleOCR และ WebAssembly Runtime สำหรับรันบนหน้าเว็บถูกบรรจุไว้ในโฟลเดอร์ `web_interface/` แล้ว สามารถรัน In-Browser OCR ในเครื่องได้ทันที 100% โดยไม่ต้องโหลดโมเดลภายนอกและไม่ต้องใช้ API Key ใด ๆ

---

## การรันและทดลองใช้งานหน้าเว็บ (Web Interface)

### 1. สตาร์ท Web Server

รันคำสั่งเปิดเครื่องบริการ (FastAPI ทำหน้าที่เสิร์ฟ Static Web Interface, ONNX Models และ API):

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend_service.app:app --host 127.0.0.1 --port 8000
```

เปิดเบราว์เซอร์ไปที่: **http://127.0.0.1:8000**

### 2. วิธีทดลองใช้งาน (How to Test)

1. **เลือกรูปภาพ**: กดเลือกเอกสารภาษาไทย/อังกฤษ (ไฟล์ PNG หรือ JPEG เช่น ประกาศราชการ, สัญญา, สลิป, หรือตารางข้อมูล)
2. **กดปุ่ม "🔍 Run In-Browser OCR"**:
   - ระบบจะประมวลผลด้วยโมเดล PaddleOCR (DBNet Detection + CRNN Recognition) ผ่าน ONNX Runtime Web ในเบราว์เซอร์ของคุณโดยตรง
   - **Privacy 100%**: รูปภาพไม่ถูกส่งไปยังเซิร์ฟเวอร์ภายนอกแม้แต่ไบต์เดียว
3. **ตรวจสอบผลลัพธ์ผ่าน Side-by-Side Workspace Layout**:
   - **ฝั่งซ้าย (Document Preview)**: แสดงภาพเอกสารพร้อมเส้นกรอบ Bounding Box แบบลอยตามสายตา (**Sticky Preview**)
   - **ฝั่งขวา (OCR Inspection Table)**: แสดงตารางผลลัพธ์ข้อความและค่าพิกัด
   - **Hover to Highlight & Auto-Scroll**: เมื่อเลื่อนเมาส์ชี้แถวใดในตาราง ภาพฝั่งซ้ายจะเลื่อน (**Smooth Scroll**) จัดตำแหน่งให้กล่องข้อความสีแดงเด่นชัดอยู่ตรงกลางสายตาทันที แม้เอกสารจะยาวหลายหน้า
4. **สลับมุมมองได้ตามต้องการ**:
   - `📄 Sentences View`: แสดงประโยค/บรรทัดเต็ม สะอาดตา อ่านง่าย
   - `🔤 Words View`: แสดงตารางแจกแจงรายคำเดี่ยว ๆ พร้อมพิกัด `(x, y, w, h)` เฉพาะของคำนั้น
5. **คัดลอกผลลัพธ์ JSON**:
   - ปุ่ม `📋 Copy Sentences + Words JSON`: ได้ JSON โครงสร้างลำดับชั้นครบถ้วน
   - ปุ่ม `📋 Copy Words Only JSON`: ได้ Array พิกัดรายคำสำหรับนำไปประมวลผลต่อ

---

## โครงสร้างผลลัพธ์ OCR (Data Structure)

ผลลัพธ์ของ In-Browser OCR ถูกออกแบบมาให้อยู่ในโครงสร้างแบบ **Hierarchical Structure** ที่เก็บทั้งระดับประโยค (สำหรับให้อ่านเข้าใจบริบทและรัน Regex) และระดับคำย่อย (สำหรับใช้ถมดำเฉพาะจุด):

### ตัวอย่าง JSON Output

```json
[
  {
    "text": "ชื่อ นายสมชาย ใจดี เบอร์โทร 081-234-5678",
    "box": {
      "x": 120,
      "y": 450,
      "width": 820,
      "height": 38
    },
    "confidence": 0.94,
    "words": [
      {
        "text": "ชื่อ",
        "box": { "x": 120, "y": 450, "width": 55, "height": 38 },
        "confidence": 0.96
      },
      {
        "text": "นาย",
        "box": { "x": 180, "y": 450, "width": 60, "height": 38 },
        "confidence": 0.95
      },
      {
        "text": "สมชาย",
        "box": { "x": 245, "y": 450, "width": 110, "height": 38 },
        "confidence": 0.93
      },
      {
        "text": "ใจดี",
        "box": { "x": 360, "y": 450, "width": 80, "height": 38 },
        "confidence": 0.92
      },
      {
        "text": "เบอร์โทร",
        "box": { "x": 450, "y": 450, "width": 120, "height": 38 },
        "confidence": 0.95
      },
      {
        "text": "081-234-5678",
        "box": { "x": 580, "y": 450, "width": 360, "height": 38 },
        "confidence": 0.97
      }
    ]
  }
]
```

---

## คำแนะนำสำหรับทีมที่นำไปทำ Pattern Filter ต่อ (Word-Level Pinpoint Redaction)

### ปัญหาของการ Redact ระดับบรรทัด (The Over-Redaction Problem)

ในการทำ Document Redaction แบบดั้งเดิม OCR จะส่งกลับมาเฉพาะกล่องข้อความระดับบรรทัด (`sentence.box`):
- หากเราตรวจพบเบอร์โทรศัพท์ `081-234-5678` ในประโยค `"เบอร์โทร 081-234-5678"`
- แล้วเรานำ `sentence.box` ไปถมดำ **ทั้งบรรทัดจะกลายเป็นแถบดำทั้งหมด** ทำให้คำว่า `"เบอร์โทร"` ซึ่งเป็นเพียง Label ปกติหายไปด้วย ส่งผลให้เอกสารเสียบริบทและอ่านไม่รู้เรื่อง (Over-redaction)

### ประโยชน์ของโครงสร้างที่เตรียมไว้ให้

ในโมดูลนี้ เราได้พัฒนาเอนจินตัดคำภาษาไทย (`Intl.Segmenter`) ควบคู่กับการคำนวณ **Proportional Glyph Bounding Boxes** (โดยคัดกรองสระและวรรณยุกต์ซ้อนแนวตั้งออก ไม่ให้เกิดปัญหากล่องเยื้อง) ทำให้ **ทุกประโยคมีพิกัดเฉพาะเจาะจงของแต่ละคำ (`words: [...]`) แนบมาด้วยเสมอ**

### วิธีนำไปใช้ใน Pattern Filter & Redactor

คนที่นำข้อมูลชุดนี้ไปทำ Pattern Filter ต่อ สามารถทำได้ง่ายดายดังนี้:

1. **Match ระดับประโยค**: ใช้ Regex / LLM ตรวจสอบ `sentence.text` เพื่อดูบริบทและจับคู่ Pattern ของ PII
2. **สกัดเฉพาะกล่องของคำที่เป็น PII**: เมื่อพบข้อความ PII ในประโยค ให้ค้นหาคำที่ตรงกันใน `sentence.words` แล้วดึงเฉพาะ `word.box` ของคำนั้นออกมา
3. **วาดกล่องดำเฉพาะจุด**: ส่งเฉพาะ `word.box` ไปยัง Redactor เพื่อถมดำ

#### ตัวอย่างการเรียกใช้ฟังก์ชัน `getRedactionBoxes`:

```javascript
import { getRedactionBoxes } from "./paddle_ocr.js";

// ตัวอย่างประโยคที่อ่านได้จาก OCR
const sentence = {
  text: "ติดต่อ นายสมชาย ใจดี โทร 081-234-5678",
  box: { x: 100, y: 200, width: 600, height: 35 },
  words: [ /* ... ข้อมูล words จาก OCR ... */ ]
};

// เมื่อ Pattern Filter ตรวจพบ PII ในบรรทัดนี้
const detectedPII = ["081-234-5678", "สมชาย"];

// เรียกฟังก์ชันดึงกล่องเฉพาะคำที่เป็น PII
const boxesToRedact = getRedactionBoxes(sentence, detectedPII);

// ผลลัพธ์: boxesToRedact จะมีเฉพาะกล่องของ "081-234-5678" และ "สมชาย"
// คำว่า "ติดต่อ", "นาย", "โทร" จะไม่ถูกถมดำ คงความสมบูรณ์ของเอกสาร 100%!
```

---

## สถาปัตยกรรมการตรวจจับ (Detection Layers)

สำหรับ Backend Pipeline ระบบรองรับการผสมผสานหลายชั้นตรวจจับ (Union Multi-Layer):

| Layer | รับผิดชอบ | ทำงานที่ | หมายเหตุ |
|---|---|---|---|
| **In-Browser PaddleOCR** | ข้อความ + พิกัดประโยคและคำรายคำ | Client Browser (ONNX WASM) | 100% Local / Zero Data Leak |
| **Pattern Classifier** | email, เบอร์โทร, บัตร ปชช.ไทย (checksum), ที่อยู่, บัญชีธนาคาร, secrets | Local Python | Source of truth เสมอ |
| **LLM Classifier** | ชื่อบุคคล, หน่วยงาน/องค์กร (Whole-page Context) | LiteLLM / Remote | วิเคราะห์บริบททั้งหน้า |
| **Presidio (Pattern-only)** | บัตรเครดิต, IP Address, IBAN, Crypto Wallet | Local Python (~110MB) | ไม่ต้องโหลดโมเดลใหญ่ |
| **detect-secrets** | AWS/GitHub/GitLab tokens, Private Keys, JWT | Local Python (~30MB) | เปิดเฉพาะ High-precision plugins |

---

## Tests

ทดสอบความถูกต้องของตรรกะการรวมประโยค, การแยกคอลัมน์, และการสับคำภาษาไทย:

```powershell
# รัน Python Unit Tests ทั้งหมด (106 tests)
.\.venv\Scripts\python.exe -m pytest tests/unit -q

# รันการทดสอบครอบคลุมทั้งหมด
.\.venv\Scripts\python.exe -m pytest -q
```

---

## ความปลอดภัยและ Privacy

- ไฟล์ `.env` (คีย์จริง) **ถูก gitignore ไว้ ไม่ขึ้น git**
- In-Browser OCR ประมวลผลบนเครื่องของผู้ใช้ทั้งหมด ข้อมูลภาพไม่รั่วไหลออกสู่อินเทอร์เน็ต
- ชุดทดสอบและโมเดล ONNX พร้อมรันแบบ Offline ได้ทันทีหลังจาก Clone โปรเจกต์

