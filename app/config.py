import os
from pydantic import BaseModel, Field


class Settings(BaseModel):
    """ตั้งค่าหลักของ API"""

    model_path: str = Field(default=os.getenv("MODEL_PATH", "models/cnn_model.onnx"))
    class_names: list[str] = Field(
        default_factory=lambda: os.getenv("CLASS_NAMES", "cat,dog").split(",")
    )
    image_size: int = Field(default=int(os.getenv("IMAGE_SIZE", "224")))
    max_file_size_mb: int = Field(default=int(os.getenv("MAX_FILE_SIZE_MB", "5")))
    num_workers: int = Field(default=int(os.getenv("NUM_WORKERS", "2")))
    preprocess_mode: str = Field(
        default=os.getenv("PREPROCESS_MODE", "rescale"),
        description="rescale = หาร 255, mobilenet_v2 = แปลงช่วงเป็น -1 ถึง 1",
    )


settings = Settings()
