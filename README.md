# PII Guardrail สำหรับรูปภาพ (POC)

Guardrail ที่สแกนรูปภาพเพื่อ **ตรวจจับและเบลอ (redact) ข้อมูลส่วนบุคคล (PII)** ก่อนที่ผู้ใช้จะอัปโหลดภาพไปยัง AI ภายนอก

Workflow:

```
image → OCR → Classifier (Pattern + LLM + Presidio + detect-secrets) → Redactor (กล่องดำ)
```

> **สถานะ: POC** — ปัจจุบัน OCR และ LLM เรียกผ่าน third-party (OCR.space / OpenRouter ผ่าน LiteLLM) เหมาะกับการทดสอบด้วยข้อมูลที่ไม่ใช่ข้อมูลจริง สำหรับ production ควรย้ายไป self-host (ดู [Roadmap](#roadmap))

---

## Requirements

- **Python 3.10+**
- **Node.js** (สำหรับรัน frontend tests เท่านั้น — ตัวเว็บเสิร์ฟผ่าน backend)
- **Docker** (สำหรับ LiteLLM proxy — เฉพาะเมื่อใช้ LLM layer)
- API keys:
  - `OCRSPACE_API_KEY` — ฟรีที่ https://ocr.space/ocrapi
  - `LITELLM_API_KEY` — จาก LiteLLM proxy ของคุณ

---

## Setup

```powershell
# 1. clone + เข้าโฟลเดอร์
git clone https://github.com/WoodMMK/pii-guardrail.git
cd pii-guardrail

# 2. สร้าง virtual environment
python -m venv .venv

# 3. ติดตั้ง dependencies (dev = ครบทุก extra: ocr, web, ner, presidio, secrets, test)
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

# 4. ตั้งค่า environment (คีย์จริง — ไฟล์นี้ถูก gitignore ไม่ขึ้น git)
Copy-Item .env.example .env
# แล้วแก้ .env ใส่ OCRSPACE_API_KEY และ LITELLM_API_KEY
```

> **หมายเหตุ:** Presidio ใช้ spaCy blank pipeline (ไม่ต้องดาวน์โหลด model). detect-secrets และ Presidio รันในเครื่อง ไม่ต้องใช้คีย์

---

## การรัน

### เปิด server (เสิร์ฟทั้งเว็บ + API)

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend_service.app:app --host 127.0.0.1 --port 8000
```

เปิดเบราว์เซอร์ที่ **http://127.0.0.1:8000**

- app โหลด key จาก `.env` อัตโนมัติ (ไม่ต้องตั้ง env ใน shell)
- **frontend กับ API เป็น server เดียวกัน** (FastAPI เสิร์ฟ static frontend ที่ `/` และ API ที่ `/api/*`) ไม่ต้องเปิด server แยก

### LiteLLM proxy (เฉพาะเมื่อใช้ LLM layer)

LLM layer ต้องการ LiteLLM proxy รันอยู่ที่ port 4000 (คนละ repo — `AI center/LiteLLM`):

```powershell
docker compose up -d litellm
```

ถ้าไม่เปิด LiteLLM ระบบยังทำงานได้ด้วย **pattern + Presidio + detect-secrets** (แค่ไม่มี LLM จับชื่อ/องค์กร)

---

## Environment variables

| ตัวแปร | จำเป็น | ค่าเริ่มต้น | คำอธิบาย |
|---|---|---|---|
| `OCRSPACE_API_KEY` | ใช่ (สำหรับ cloud OCR) | `helloworld` (ถูก throttle) | คีย์ OCR.space |
| `LITELLM_API_KEY` | สำหรับ LLM layer | — | Bearer key ของ LiteLLM proxy |
| `LITELLM_BASE_URL` | ไม่ | `http://localhost:4000/v1` | endpoint ของ LiteLLM |
| `PII_GUARDRAIL_LLM_MODEL` | ไม่ | `Model_B` | ชื่อ model ที่จะเรียก (ควรเป็น non-reasoning) |
| `OCRSPACE_LANGUAGE` | ไม่ | `tha` | ภาษาที่ให้ OCR.space อ่าน |
| `PII_GUARDRAIL_OCR_BACKEND` | ไม่ | `ocrspace` | ตั้ง `paddle` เพื่อใช้ PaddleOCR local แทน |

ดูตัวอย่างครบใน [`.env.example`](.env.example)

---

## Architecture

### Detection layers (ผลรวมกันแบบ union)

| Layer | รับผิดชอบ | ทำงานที่ | หมายเหตุ |
|---|---|---|---|
| **Pattern** (regex เขียนเอง) | email, เบอร์, เงิน, วันที่, บัตร ปชช.ไทย (checksum), ที่อยู่ไทย, ทะเบียนรถ, บัญชีธนาคาร, secret | local | source of truth เสมอ |
| **LLM** (LiteLLM / OpenRouter) | ชื่อคน, องค์กร, ที่อยู่ (whole-page, ดูบริบททั้งหน้า) | remote | เข้าใจ OCR ที่เพี้ยน |
| **Presidio** (pattern-only, ไม่มี NER) | credit card, IP, IBAN, crypto wallet | local (~110MB) | blank spaCy, ไม่โหลด model |
| **detect-secrets** (Yelp) | AWS/GitHub/GitLab keys, JWT, private key ฯลฯ | local (~30MB) | เปิดเฉพาะ plugin แม่นสูง (ปิด entropy) |

> **หมายเหตุ:** ชื่อคน/องค์กร ถูกยกให้ **LLM รับผิดชอบเท่านั้น** — pattern ไม่จับชื่ออีกแล้ว (เดิมทำ false-positive กับหัวข้อเช่น "Tuition Fee")

### OCR

- ค่าเริ่มต้น: **OCR.space** (cloud, ~1.8s/ภาพ, อ่านไทย+อังกฤษ, คืน bounding box)
- ทางเลือก: **PaddleOCR** local (ตั้ง `PII_GUARDRAIL_OCR_BACKEND=paddle`) — แม่นกว่าแต่ช้ามากบน CPU

### ส่วนประกอบหลัก (`pii_guardrail/`)

| ไฟล์ | หน้าที่ |
|---|---|
| `pipeline.py` | orchestrate: preprocess → OCR → detect → redact |
| `detector.py` | pattern classifier (`classify_segment`) + `Detector` |
| `composite.py` | รวมหลาย classifier layer (union + per-source breakdown) |
| `litellm_backend.py` | LLM classifier (whole-page) |
| `presidio_pattern_backend.py` | Presidio pattern-only classifier |
| `secrets_backend.py` | detect-secrets classifier |
| `ocrspace_backend.py` / `paddle_backend.py` | OCR backends |
| `redactor.py` | วาดกล่องดำทับ region |
| `preprocessor.py` | denoise / (optional) upscale+deskew / quality gate |
| `models.py` | data models + `SensitiveCategory` enum |

Web adapter อยู่ใน `backend_service/` (FastAPI) และ frontend ใน `web_interface/`

---

## Debug view

หน้าเว็บมี panel **"OCR debug"** (พับได้ ใต้ผลลัพธ์) แสดงทุก segment ที่ OCR อ่านได้ พร้อม:

- ข้อความที่อ่านได้ + bounding box + confidence
- **Classified as (by layer)** — แต่ละ classifier (pattern / llm / presidio / detect-secrets) จับ segment นั้นเป็นอะไร
- **Redacted?** — segment นั้นถูกเบลอไหม

ช่วย debug ว่าถ้า segment ไหนจับผิด/พลาด เป็นความผิดของ layer ไหน

---

## Tests

```powershell
# Python unit tests (เร็ว)
.\.venv\Scripts\python.exe -m pytest tests/unit -q

# ทั้งหมด (รวม property-based — ช้ากว่า)
.\.venv\Scripts\python.exe -m pytest -q

# Frontend tests
cd web_interface
npx vitest --run
```

---

## Git branches

| Branch | เนื้อหา |
|---|---|
| `master` | Baseline: OCR.space + LiteLLM Model_B |
| `feature/pattern-filters` | + Presidio + detect-secrets + `.env` + per-source debug (branch หลักปัจจุบัน) |
| `experiment/screenpipe-detector` | ทดลอง screenpipe image-detector (พักไว้ — ไม่เหมาะกับเอกสารไทย) |

---

## Roadmap (POC → Production)

- **Privacy:** ย้าย OCR + LLM ไป self-host (LLM บนการ์ดจอบริษัท + OCR ในเครื่อง) เพื่อไม่ให้ข้อมูลออกนอกองค์กร
- **Client-side OCR:** กำลังพิจารณา Tesseract.js (รัน OCR ใน browser ภาพไม่ออกจากเครื่อง user)
- **License:** ถ้าจะใช้ screenpipe model เชิงพาณิชย์ ต้องซื้อ license (CC BY-NC 4.0)

---

## หมายเหตุด้านความปลอดภัย

- `.env` (คีย์จริง) **ถูก gitignore ไม่ขึ้น git** — แชร์คีย์ผ่านช่องทางปลอดภัย ไม่ commit
- LiteLLM proxy ในโปรเจกต์นี้ถูกถอด PII guardrail ของตัวเองออก (เพราะ app นี้คือ guardrail ที่ต้องเห็นข้อมูลเต็มเพื่อ classify) — ถ้ามีคนอื่นใช้ proxy ร่วม ควรพิจารณาเปิดกลับ
