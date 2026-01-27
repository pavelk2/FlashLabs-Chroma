FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

WORKDIR /app

# System dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Pre-download model (optional: mount or bake in)
# ENV CHROMA_MODEL_ID=FlashLabs/Chroma-4B
# RUN python -c "from transformers import AutoModelForCausalLM, AutoProcessor; \
#     AutoModelForCausalLM.from_pretrained('FlashLabs/Chroma-4B', trust_remote_code=True); \
#     AutoProcessor.from_pretrained('FlashLabs/Chroma-4B', trust_remote_code=True)"

ENV HOST=0.0.0.0
ENV PORT=8000
EXPOSE 8000

CMD ["python", "server.py"]
