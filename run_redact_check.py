import io, base64, json, urllib.request, urllib.error
import numpy as np
from PIL import Image
from tests.integration.sample_corpus import render_scanned_sample

# สร้างรูปสแกน >=150 DPI ที่มี PII (อีเมล) เพื่อให้ผ่าน quality gate และตรวจเจอ
img_bgr = render_scanned_sample("user@example.com")
# render_scanned_sample คืน BGR -> แปลงกลับเป็น RGB เพื่อ save เป็น PNG ให้ backend อ่าน
rgb = img_bgr[:, :, ::-1]
buf = io.BytesIO()
Image.fromarray(rgb, "RGB").save(buf, format="PNG")
png_bytes = buf.getvalue()
print("upload PNG size:", len(png_bytes), "bytes; shape:", img_bgr.shape)

# ประกอบ multipart/form-data เหมือน frontend (field ชื่อ image)
boundary = "----pii-test-boundary"
body = b""
body += ("--" + boundary + "\r\n").encode()
body += b'Content-Disposition: form-data; name="image"; filename="scan.png"\r\n'
body += b"Content-Type: image/png\r\n\r\n"
body += png_bytes + b"\r\n"
body += ("--" + boundary + "--\r\n").encode()

req = urllib.request.Request(
    "http://127.0.0.1:8077/api/redact",
    data=body,
    headers={"Content-Type": "multipart/form-data; boundary=" + boundary},
    method="POST",
)
try:
    with urllib.request.urlopen(req, timeout=180) as resp:
        status = resp.status
        payload = json.loads(resp.read().decode())
except urllib.error.HTTPError as e:
    print("HTTP error", e.code, e.read().decode()[:500]); raise SystemExit(1)

print("HTTP", status)
dr = payload["detection_result"]
print("count =", dr["count"])
print("image dims =", dr["image_width"], "x", dr["image_height"])
for r in dr["regions"]:
    print("  region:", r["categories"], "box", r["box"], "conf", round(r["confidence"],3))
print("quality_sufficient =", payload["quality_sufficient"])
print("warnings =", payload["warnings"])
print("redacted_image format =", payload["redacted_image"]["format"], "base64 len =", len(payload["redacted_image"]["base64"]))

# ถอด base64 -> เซฟไฟล์ + ตรวจว่ากล่องที่ตรวจเจอถูกทาดำจริง
red_bytes = base64.b64decode(payload["redacted_image"]["base64"])
open("redacted_result.png", "wb").write(red_bytes)
red = np.asarray(Image.open(io.BytesIO(red_bytes)).convert("RGB"))
print("saved redacted_result.png; decoded shape:", red.shape)
all_black = True
for r in dr["regions"]:
    b = r["box"]
    sub = red[b["y"]:b["y"]+b["height"], b["x"]:b["x"]+b["width"]]
    black = bool((sub == 0).all())
    print("  box", (b["x"],b["y"],b["width"],b["height"]), "fully black =", black)
    all_black = all_black and black
print("ALL detected boxes fully black =", all_black)
