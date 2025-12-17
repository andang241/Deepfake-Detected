FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

ENV HF_HOME=/models/hf
ENV TRANSFORMERS_CACHE=/models/hf
ENV HF_HUB_DISABLE_TELEMETRY=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download model at build time (image sẽ to hơn nhưng runtime không cần internet)
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='prithivMLmods/Deep-Fake-Detector-v2-Model', local_dir='/models/hf', local_dir_use_symlinks=False)"

# Để transformers tìm đúng local cache khi load bằng MODEL_ID
ENV HF_HUB_OFFLINE=1

COPY app.py .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host=0.0.0.0", "--port=8000"]