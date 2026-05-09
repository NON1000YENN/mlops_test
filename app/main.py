from __future__ import annotations

import asyncio
from concurrent.futures import ProcessPoolExecutor

from fastapi import FastAPI, File, HTTPException, UploadFile, status

from app.config import settings
from app.inference import ImageDecodeError, predict_from_bytes
from app.schemas import HealthResponse, PredictionResponse

app = FastAPI(
    title="Cat vs Dog Image Classification API",
    description="FastAPI + ONNX Runtime สำหรับจำแนกรูปภาพแมว/สุนัข",
    version="1.0.0",
)

executor = ProcessPoolExecutor(max_workers=settings.num_workers)


@app.on_event("shutdown")
def shutdown_event() -> None:
    executor.shutdown(wait=False, cancel_futures=True)


@app.get("/", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(status="ok", model=settings.model_path)


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """
    รับไฟล์รูปภาพ แล้วคืนผลทำนายเป็น JSON
    - ใช้ async def ตามโจทย์
    - ส่งงาน inference ไปที่ ProcessPoolExecutor เพื่อกัน API ค้างจากงาน CPU-bound
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="กรุณาส่งไฟล์รูปภาพเข้ามา",
        )

    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="รองรับเฉพาะไฟล์รูปภาพเท่านั้น เช่น jpg, jpeg, png",
        )

    image_bytes = await file.read()
    max_bytes = settings.max_file_size_mb * 1024 * 1024

    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ไฟล์ว่างเปล่า",
        )

    if len(image_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"ไฟล์ใหญ่เกินไป จำกัดไม่เกิน {settings.max_file_size_mb} MB",
        )

    loop = asyncio.get_running_loop()

    try:
        result = await loop.run_in_executor(executor, predict_from_bytes, image_bytes)
    except ImageDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ไฟล์รูปภาพเสีย หรือเปิดอ่านไม่ได้",
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="ไม่พบไฟล์โมเดลบน Server",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"เกิดข้อผิดพลาดระหว่างทำนายผล: {type(exc).__name__}",
        ) from exc

    return PredictionResponse(filename=file.filename, **result)
