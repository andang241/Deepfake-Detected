import os
import io
from fastapi import FastAPI, UploadFile, File
from PIL import Image, ImageOps
import torch
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)

# ====== CONFIG ======
MODEL_PATH = "/models/hf"   # nơi snapshot_download đã lưu model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Root path để chạy sau reverse proxy (Run:AI gateway)
# Local chạy bình thường vì mặc định ROOT_PATH=""
ROOT_PATH = os.getenv("ROOT_PATH", "")

# (Khuyến nghị) In log ra để debug
print(f"[INFO] Using device: {DEVICE}")
print(f"[INFO] Loading model from: {MODEL_PATH}")
print(f"[INFO] ROOT_PATH: {ROOT_PATH!r}")

# ====== LOAD MODEL ======
processor = AutoImageProcessor.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

model = AutoModelForImageClassification.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
).to(DEVICE)

model.eval()

# ====== FASTAPI ======
app = FastAPI(
    title="Deepfake Detector Inference",
    root_path=ROOT_PATH,
)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "root_path": ROOT_PATH,
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()

    # xử lý EXIF orientation để tránh ảnh bị xoay sai
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    image = ImageOps.exif_transpose(image)

    inputs = processor(
        images=image,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]

    probs = probs.detach().cpu().tolist()
    pred_id = int(torch.argmax(logits, dim=-1).item())

    return {
        "predicted_id": pred_id,
        "predicted_label": model.config.id2label.get(pred_id, str(pred_id)),
        "probabilities": {
            model.config.id2label.get(i, str(i)): float(p)
            for i, p in enumerate(probs)
        },
    }
