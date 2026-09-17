# PII Guardrail สำหรับรูปภาพ — สรุปงาน

## เป้าหมาย
POC ระบบ guardrail ที่สแกนรูปภาพเพื่อกันไม่ให้ผู้ใช้ทั่วไปอัปโหลดภาพที่มีข้อมูล sensitive (PII) ไปยัง AI ภายนอกที่บริษัทเช่าใช้

## Workflow
```
image → OCR (OCR.space) → Classifier (Pattern + LLM) → Redactor (กล่องดำ)
```

---

## ตัวอย่าง Data Flow ด้วยข้อมูลจริง

ทดสอบด้วยรูปที่มีข้อความ 3 บรรทัด: ชื่อคน, อีเมล, จำนวนเงิน

### 🔹 ขั้นที่ 1 — OCR.space

**Input:** ไฟล์รูป PNG (binary ~11 KB) ส่งแบบ `multipart/form-data` พร้อม `language=tha`, `OCREngine=2`, `isOverlayRequired=true`

**Output (raw JSON จาก OCR.space):** — คืนข้อความเป็น "บรรทัด" โดยแต่ละบรรทัดมี "คำ" ย่อย พร้อมพิกัดของทุกคำ
```json
{
  "LineText": "นายสมชาย ใจดี",
  "Words": [
    { "WordText": "นาย",  "Left": 20,  "Top": 36, "Width": 46, "Height": 24 },
    { "WordText": "สม",   "Left": 67,  "Top": 36, "Width": 30, "Height": 24 },
    { "WordText": "ชาย",  "Left": 97,  "Top": 36, "Width": 47, "Height": 24 },
    { "WordText": "ใจดี", "Left": 146, "Top": 36, "Width": 42, "Height": 24 }
  ]
}
```
> OCR.space แยกเป็น "คำ" (word-level) แต่ละคำมีกล่องของตัวเอง

**หลังระบบเราแปลง (รวมคำเป็นบรรทัด + กล่องเดียว):**

| # | ข้อความ | กล่อง (x, y, w, h) |
|---|---|---|
| 0 | นายสมชาย ใจดี | 20, 36, 168, 24 |
| 1 | อีเมล somchai@example.com | 20, 84, 332, 32 |
| 2 | ยอดชำระ 1,500.00 บาท | 20, 136, 262, 28 |

### 🔹 ขั้นที่ 2 — LLM (Model_B ผ่าน LiteLLM)

**Input:** ส่งข้อความทั้งหน้าให้ครั้งเดียว พร้อมเลขกำกับบรรทัด
```
Segments:
[0] นายสมชาย ใจดี
[1] อีเมล somchai@example.com
[2] ยอดชำระ 1,500.00 บาท
```

**Output (raw จาก LLM):** — ตอบเป็น JSON ล้วน บอกว่าบรรทัดไหนเป็น PII ประเภทอะไร
```json
{"0":["person_name"],"1":["email"],"2":["money_amount"]}
```

### 🔹 ขั้นที่ 3 — Classifier (Pattern + LLM รวมกัน)

Pattern classifier (regex ที่เราเขียนเอง) **ยังทำงานอยู่เสมอ** รันคู่กับ LLM ทุกครั้ง แล้วเอาผลมารวมกัน (union)

| # | ข้อความ | Pattern จับได้ | LLM จับได้ | ผลรวมสุดท้าย |
|---|---|---|---|---|
| 0 | นายสมชาย ใจดี | person_name | person_name | **person_name** |
| 1 | อีเมล somchai@example.com | email, url | email | **email, url** |
| 2 | ยอดชำระ 1,500.00 บาท | money_amount | money_amount | **money_amount** |

### 🔹 ขั้นที่ 4 — Redactor
เอา "กล่อง" จากขั้น OCR มาวาดกล่องดำทับเฉพาะบรรทัดที่มีผลลัพธ์ (ทั้ง 3 บรรทัดในเคสนี้เป็น PII จึงถูกปิดหมด)

---

## Pattern กับ LLM ซ้ำซ้อนกันไหม? (Fail-safe design)

**ซ้ำได้ แต่ไม่เป็นปัญหา** เพราะผลลัพธ์เป็น `set` แล้วรวมด้วย **union** — สมาชิกซ้ำจะยุบเหลือตัวเดียวอัตโนมัติ

- แถว 0, 2: pattern กับ LLM ตอบตรงกัน → ยุบเหลือตัวเดียว (ไม่ซ้ำ)
- แถว 1: **เสริมกัน** — pattern จับ `url` เพิ่มที่ LLM มองข้าม, LLM จับ `person_name` ที่ pattern ทำไม่ได้

**ทำไม design แบบนี้ถึงดี:**
- ซ้ำ = ไม่เสียหาย (set จัดการเอง, กล่องดำวาดครั้งเดียวต่อบรรทัดอยู่แล้ว)
- เสริมกัน = ครอบคลุมกว่าใช้ตัวเดียว
- ปลอดภัยไว้ก่อน — สำหรับ guardrail จับ**เกิน**ดีกว่าจับ**ขาด** ถ้าตัวใดตัวหนึ่งจับได้ก็ redact เลย

---

## ทำไมถึงมาจบที่ LLM (ไทม์ไลน์ classifier ที่ลองมา)

| Classifier | สถานะ | เหตุผล |
|---|---|---|
| Pattern (regex เขียนเอง) | ✅ ใช้อยู่ | เบา, เสถียร, จับข้อมูลรูปแบบชัดได้ดี |
| PyThaiNLP NER (local CRF) | ❌ เลิก | แม่นน้อย จับ OCR เพี้ยนไม่ได้ |
| Presidio | ❌ เลิก | กิน RAM ~600MB เครื่องรับไม่ไหว |
| thainer-corpus-v2 (HF API) | ❌ เลิก | endpoint ฟรีล่มเป็นช่วงๆ (ไม่เสถียร) |
| **LiteLLM Model_B** | ✅ **ใช้อยู่** | เบา (compute อยู่ที่ cloud), เสถียร, จับ OCR เพี้ยนได้ดีสุด |

### ทำไมไม่รัน thainer-v2 ในเครื่องเอง
เพราะ v2 เป็น **Transformer (BERT)** ไม่ใช่ CRF ตัวเบา:

| | PyThaiNLP thainer (CRF) | thainer-corpus-v2 (Transformer) |
|---|---|---|
| ขนาด model | ~1-2 MB | ~400-500 MB |
| dependency | เบา | ต้อง PyTorch (~500MB-1GB+) |
| RAM ตอนรัน | เล็กน้อย | ~1-1.5 GB |

> รันเองจะหนักกว่า Presidio (ที่เครื่องรับไม่ไหว) เสียอีก จึงไม่รันเอง

### ทำไม HF API ถึงล่มเป็นบางครั้ง
- HF Inference ฟรีเป็น **shared / best-effort** ไม่มี uptime guarantee สำหรับ community model
- model ตัวเล็กที่คนใช้น้อยจะไม่ถูก keep-warm — เวลา HF backend มีปัญหา จะโดนกระทบก่อน (โผล่มาเป็น HTTP 500)
- **ไม่ใช่ที่ตัว model หรือ payload เรา** — ยิงซ้ำภายหลังกลับมา 200 ปกติ แต่เชื่อถือสำหรับ production ไม่ได้
- ถ้าจะใช้จริงต้องทำ Dedicated Endpoint (เสียเงิน host) → ต้นทุนพอๆ กับ self-host

---

## รายละเอียดเชิงเทคนิค

### OCR — OCR.space API (cloud)
- เร็ว ~1.8 วินาที/ภาพ, อ่านไทย+อังกฤษ, คืน bounding box ระดับคำ
- แทน PaddleOCR local ที่ช้า ~80 วินาที/ภาพ บน CPU
- ปิด auto-upscale + deskew → กล่องดำไม่เพี้ยน + เร็วขึ้น

### Classifier
- **Pattern (local, regex เขียนเอง — ไม่ใช่ ML):** email, เบอร์โทร, บัตรเครดิต (Luhn), เลขบัตรประชาชนไทย (checksum), URL, เงิน, วันที่, API key / secret, job title
- **LLM whole-page (Model_B ผ่าน OpenRouter):** ส่งทั้งหน้าครั้งเดียว ตอบ JSON ว่าบรรทัดไหน sensitive → จับชื่อคน/องค์กร/ที่อยู่ที่ pattern ทำไม่ได้
- ผล union กัน โดย pattern เป็น source of truth

### LLM (ผ่าน LiteLLM proxy)
- ใช้ **Model_B** (non-reasoning, เร็ว, ตอบ JSON ตรง)
- reasoning models ใช้ไม่ได้ — เผา token ไปกับการคิดจนตอบไม่ทัน
- ถอด PII guardrail ของ LiteLLM ออก เพราะบล็อก request (app เราเองคือ guardrail ที่ต้องเห็นข้อมูลเต็ม)

### Web UI
- อัปโหลดภาพ → แสดงภาพที่ redact แล้ว + ดาวน์โหลด
- **OCR debug panel:** ตารางแสดงทุกบรรทัด พร้อม ประเภท PII / Redacted? / Confidence / ตำแหน่ง

## ผลการทดสอบ
- จับชื่อคนที่ OCR อ่านเพี้ยนได้ (เคสที่ NER แบบเก่าพลาด เช่น `ธีตรัตน์`, `บั่วบุญมี`)
- Unit tests 96/96, frontend tests 13/13

## ข้อควรทำต่อ (POC → Production)
- **Privacy:** OCR + LLM ตอนนี้เป็น third-party (ข้อมูลออกนอกองค์กร) เหมาะกับ POC เท่านั้น — production ควรย้ายไป **LLM บนการ์ดจอบริษัท** + self-host OCR เพื่อไม่ให้ข้อมูลรั่ว
- พิจารณาเปิด PII guardrail ของ LiteLLM กลับ ถ้ามีคนอื่นใช้ proxy ร่วม
