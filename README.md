# Cat vs Dog Image Classification API

โปรเจกต์ส่วนที่ 2: FastAPI + Production Error Handling + ONNX Runtime

## โครงสร้างโปรเจกต์

```text
app/
  main.py          # FastAPI endpoint
  inference.py     # โหลดโมเดล ONNX + preprocess + predict
  config.py        # ตั้งค่า API
  schemas.py       # Pydantic response schema
models/
  cnn_model.onnx   # โมเดลที่เพื่อนแปลงเป็น ONNX แล้ว
tests/
  test_api.py      # pytest สำหรับ API
Dockerfile
requirements.txt
```

## วิธีรันบนเครื่อง

```bash
python -m venv .venv
source .venv/bin/activate  # Windows ใช้ .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

เปิดหน้าเอกสาร API:

```text
http://127.0.0.1:8000/docs
```

## ทดสอบเรียก API ด้วย cURL

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@sample.jpg"
```

ตัวอย่างผลลัพธ์:

```json
{
  "filename": "sample.jpg",
  "label": "cat",
  "confidence": 0.93,
  "probabilities": {
    "cat": 0.93,
    "dog": 0.07
  }
}
```

> หมายเหตุ: ถ้าผลทำนายสลับ cat/dog ให้แก้ตัวแปร `CLASS_NAMES` เป็น `dog,cat`

## Environment Variables

| ชื่อ | ค่า default | ความหมาย |
|---|---|---|
| MODEL_PATH | models/cnn_model.onnx | path ของโมเดล ONNX |
| CLASS_NAMES | cat,dog | ชื่อ class ตามลำดับ output ของโมเดล |
| PREPROCESS_MODE | rescale | `rescale` = /255, `mobilenet_v2` = [-1,1] |
| NUM_WORKERS | 2 | จำนวน process สำหรับ inference |
| MAX_FILE_SIZE_MB | 5 | จำกัดขนาดไฟล์รูป |

## รันด้วย Docker

```bash
docker build -t catdog-api .
docker run -p 7860:7860 catdog-api
```

เรียก API:

```bash
curl -X POST "http://127.0.0.1:7860/predict" \
  -F "file=@sample.jpg"
```

## รัน test

```bash
pytest -q
```

## Error Handling ที่ทำไว้

- ไม่ส่งไฟล์ → 400 Bad Request
- ไฟล์ไม่ใช่รูปภาพ → 400 Bad Request
- ไฟล์รูปภาพเสีย/อ่านไม่ได้ → 400 Bad Request
- ไฟล์ว่าง → 400 Bad Request
- ไฟล์ใหญ่เกินกำหนด → 413 Request Entity Too Large
- ไม่พบโมเดลบน server → 500 Internal Server Error

