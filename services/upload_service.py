"""Upload Service — owns raw image files on disk.

POST /upload persists the file, then publishes ``image.uploaded`` with
the file path + id.  Binary bytes stay off the bus.

Publishes:  image.uploaded
Subscribes: —
"""

from __future__ import annotations

import os
import shutil
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from events import IMAGE_UPLOADED, ImageUploadedPayload, make_event
from messaging import MessageBus

STORAGE_DIR = os.getenv("UPLOAD_STORAGE_DIR", "./data/uploads")

app = FastAPI(title="Upload Service")

_bus: MessageBus | None = None


def set_bus(bus: MessageBus) -> None:
    global _bus
    _bus = bus


def get_bus() -> MessageBus:
    if _bus is None:
        raise RuntimeError("Upload service bus is not configured")
    return _bus


class UploadResponse(BaseModel):
    image_id: str
    file_path: str
    file_size_bytes: int
    mime_type: str


@app.post("/upload", response_model=UploadResponse)
def upload_image(file: UploadFile = File(...)) -> UploadResponse:
    os.makedirs(STORAGE_DIR, exist_ok=True)

    image_id = str(uuid.uuid4())
    suffix = "jpg"
    if file.filename and "." in file.filename:
        suffix = file.filename.rsplit(".", 1)[-1]
    dest_path = os.path.join(STORAGE_DIR, f"{image_id}.{suffix}")

    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    file_size = os.path.getsize(dest_path)
    mime = file.content_type or "image/jpeg"

    payload = ImageUploadedPayload(
        image_id=image_id,
        file_path=dest_path,
        file_size_bytes=file_size,
        mime_type=mime,
    )
    get_bus().publish(IMAGE_UPLOADED, make_event(IMAGE_UPLOADED, payload))

    return UploadResponse(
        image_id=image_id,
        file_path=dest_path,
        file_size_bytes=file_size,
        mime_type=mime,
    )


def register(bus: MessageBus) -> None:
    """Wire this service to a bus.  Upload Service publishes only."""
    set_bus(bus)


if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    from messaging import make_default_bus

    register(make_default_bus())
    uvicorn.run(app, host="0.0.0.0", port=8001)
