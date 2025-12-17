import io
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_ID = "prithivMLmods/Deep-Fake-Detector-v2-Model"

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoImageProcessor.from_pretrained(MODEL_ID, local_files_only=True)
model = AutoModelForImageClassification.from_pretrained(MODEL_ID, local_files_only=True).to(device)
model.eval()

@app.get("/health")
def health():
    return {"status": "ok", "device": device}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].detach().cpu().tolist()
        pred = int(torch.argmax(logits, dim=-1).item())

    # trả cả label + score để dễ kiểm tra mapping
    return {
        "pred_id": pred,
        "pred_label": model.config.id2label.get(pred, str(pred)),
        "scores": {model.config.id2label.get(i, str(i)): float(p) for i, p in enumerate(probs)},
    }
