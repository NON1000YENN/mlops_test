from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def make_test_image() -> BytesIO:
    img = Image.new(
        "RGB",
        (224, 224),
        color=(255, 255, 255)
    )

    buf = BytesIO()

    img.save(buf, format="JPEG")

    buf.seek(0)

    return buf


def test_health_check():

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"


def test_predict_valid_image():

    image = make_test_image()

    response = client.post(
        "/predict",
        files={
            "file": (
                "test.jpg",
                image,
                "image/jpeg"
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "label" in data
    assert "confidence" in data
    assert "probabilities" in data


def test_predict_non_image_file():

    response = client.post(
        "/predict",
        files={
            "file": (
                "test.txt",
                BytesIO(b"hello"),
                "text/plain"
            )
        },
    )

    assert response.status_code == 400


def test_predict_empty_image():

    response = client.post(
        "/predict",
        files={
            "file": (
                "empty.jpg",
                BytesIO(b""),
                "image/jpeg"
            )
        },
    )

    assert response.status_code == 400


def test_predict_corrupted_image():

    response = client.post(
        "/predict",
        files={
            "file": (
                "broken.jpg",
                BytesIO(b"not really image"),
                "image/jpeg"
            )
        },
    )

    assert response.status_code in [400, 500]

