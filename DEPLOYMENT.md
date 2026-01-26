# Deployment Guide

This guide covers deploying FlashLabs Chroma as a web service.

## Quick Start Options

| Platform | Difficulty | GPU Cost | Best For |
|----------|------------|----------|----------|
| Hugging Face Spaces | Easy | ~$1.05/hr (A10G) | Demos, prototypes |
| Replicate | Easy | Pay-per-second | Production APIs |
| RunPod | Medium | ~$0.44/hr (RTX 4090) | Cost-effective |
| Docker (self-hosted) | Medium | Your hardware | Full control |

---

## Option 1: Hugging Face Spaces (Recommended for Demos)

The easiest way to deploy with a web UI.

### Steps

1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces)

2. Select **Gradio** as the SDK

3. Choose **A10G Large** hardware (or T4 for testing)

4. Upload these files:
   - `app_gradio.py` (rename to `app.py`)
   - `requirements.txt`
   - `example/` folder (for voice cloning)
   - `chroma/` folder

5. The Space will automatically build and deploy

### Using the Spaces YAML Config

Copy `README_SPACES.md` to your Space as `README.md` to configure settings.

---

## Option 2: Replicate (Recommended for APIs)

Best for production API access with pay-per-use pricing.

### Steps

1. Install Cog: `pip install cog`

2. Create `cog.yaml`:
```yaml
build:
  gpu: true
  cuda: "12.6"
  python_version: "3.11"
  python_packages:
    - torch==2.7.1
    - transformers==5.0.0rc0
    - accelerate>=1.7.0
    - soundfile>=0.13.0
    - librosa>=0.11.0

predict: "predict.py:Predictor"
```

3. Create `predict.py` with your model inference code

4. Push to Replicate:
```bash
cog login
cog push r8.im/your-username/chroma
```

---

## Option 3: Docker (Self-Hosted)

For running on your own GPU server.

### Prerequisites

- NVIDIA GPU with 16GB+ VRAM
- Docker with NVIDIA Container Toolkit
- CUDA 12.6 compatible driver

### Build and Run

```bash
# Build the image
docker build -t chroma-api .

# Run with GPU
docker run --gpus all -p 8000:8000 chroma-api
```

### Using Docker Compose

```bash
docker compose up --build
```

The API will be available at `http://localhost:8000`.

---

## Option 4: RunPod

Cost-effective GPU cloud with hourly billing.

### Steps

1. Go to [runpod.io](https://runpod.io) and create an account

2. Deploy a **GPU Pod** with:
   - Template: PyTorch 2.x
   - GPU: RTX 4090 or A100
   - Disk: 50GB+

3. SSH into the pod and clone this repo

4. Install dependencies and run:
```bash
pip install -r requirements.txt
python app.py
```

5. Expose port 8000 through RunPod's proxy

---

## API Reference

### Endpoints

#### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda:0",
  "cuda_available": true
}
```

#### `GET /speakers`
List available voice styles.

**Response:**
```json
{
  "speakers": ["scarlett_johansson", "donald_trump", "ariana_grande", "lebron_james"]
}
```

#### `POST /chat`
Generate audio response from audio input.

**Request:**
```json
{
  "audio_base64": "<base64-encoded-wav>",
  "speaker": "scarlett_johansson",
  "max_new_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9
}
```

**Response:**
```json
{
  "audio_base64": "<base64-encoded-wav>",
  "sample_rate": 24000
}
```

#### `POST /chat/upload`
Multipart form upload for easier testing.

**Form Fields:**
- `audio`: Audio file (WAV recommended)
- `speaker`: Voice style
- `max_new_tokens`: Max generation length
- `temperature`: Sampling temperature

**Response:** WAV audio file download

---

## Example Client Code

### Python

```python
import base64
import requests

# Read audio file
with open("input.wav", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode()

# Send request
response = requests.post(
    "http://localhost:8000/chat",
    json={
        "audio_base64": audio_base64,
        "speaker": "scarlett_johansson",
        "max_new_tokens": 100
    }
)

# Save response audio
audio_bytes = base64.b64decode(response.json()["audio_base64"])
with open("response.wav", "wb") as f:
    f.write(audio_bytes)
```

### cURL

```bash
# Using file upload endpoint
curl -X POST http://localhost:8000/chat/upload \
  -F "audio=@input.wav" \
  -F "speaker=scarlett_johansson" \
  --output response.wav
```

### JavaScript

```javascript
const audioBlob = await fetch('input.wav').then(r => r.blob());
const base64 = await blobToBase64(audioBlob);

const response = await fetch('http://localhost:8000/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    audio_base64: base64,
    speaker: 'scarlett_johansson'
  })
});

const { audio_base64 } = await response.json();
playBase64Audio(audio_base64);
```

---

## Hardware Requirements

| Configuration | VRAM | Performance |
|--------------|------|-------------|
| Minimum | 8GB | Slow, may OOM on long inputs |
| Recommended | 16GB | Good performance |
| Optimal | 24GB+ | Best performance, longer contexts |

**Tested GPUs:**
- NVIDIA RTX 4090 (24GB) - Excellent
- NVIDIA A10G (24GB) - Excellent
- NVIDIA A100 (40/80GB) - Excellent
- NVIDIA T4 (16GB) - Adequate
- NVIDIA RTX 3090 (24GB) - Good

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `FlashLabs/Chroma-4B` | HuggingFace model ID or local path |
| `PORT` | `8000` | Server port |
| `HF_HOME` | `~/.cache/huggingface` | Model cache directory |

---

## Troubleshooting

### Out of Memory (OOM)
- Reduce `max_new_tokens`
- Use a GPU with more VRAM
- Enable `torch.cuda.empty_cache()` between requests

### Model Loading Slow
- First load downloads ~8GB of weights
- Use a volume mount to cache weights between restarts

### Audio Quality Issues
- Ensure input audio is clear and 16kHz+ sample rate
- WAV format is recommended over MP3
