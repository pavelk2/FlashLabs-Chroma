# Deploying Chroma as a Voice Agent Web Service

## Quick Start

### Local (with GPU)

```bash
pip install -r requirements.txt
python server.py
```

The service starts at `http://localhost:8000`. Open it in a browser to use the voice agent UI.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `CHROMA_MODEL_ID` | `FlashLabs/Chroma-4B` | HuggingFace model ID or local path |
| `CHROMA_DEVICE` | `auto` | Device map (`auto`, `cuda:0`, `cpu`) |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | Bind port |

### Docker

```bash
# Build
docker build -t chroma-voice-agent .

# Run (with GPU passthrough)
docker run --gpus all -p 8000:8000 \
  -e CHROMA_MODEL_ID=FlashLabs/Chroma-4B \
  chroma-voice-agent
```

To use a pre-downloaded model, mount it as a volume:

```bash
docker run --gpus all -p 8000:8000 \
  -v /path/to/Chroma-4B:/models/Chroma-4B \
  -e CHROMA_MODEL_ID=/models/Chroma-4B \
  chroma-voice-agent
```

## API Reference

### `GET /api/health`

Health check. Returns model loading status.

### `GET /api/speakers`

Lists available speaker voices for cloning.

### `POST /api/generate`

Generate a voice response from audio input.

**Form parameters:**
- `audio` (file, required) — Input audio (WAV, WebM, MP3, etc.)
- `speaker` (string) — Speaker voice ID (default: `scarlett_johansson`)
- `max_new_tokens` (int) — Max audio tokens (default: `200`)
- `temperature` (float) — Sampling temperature (default: `0.7`)
- `top_k` (int) — Top-k sampling (default: `50`)
- `top_p` (float) — Nucleus sampling (default: `0.9`)

**Returns:** `audio/wav` response.

```bash
curl -X POST http://localhost:8000/api/generate \
  -F "audio=@input.wav" \
  -F "speaker=scarlett_johansson" \
  --output response.wav
```

### `WebSocket /ws/voice`

Real-time voice interaction via WebSocket.

**Client sends:**
```json
{"type": "config", "speaker": "scarlett_johansson", "max_new_tokens": 200}
{"type": "audio", "data": "<base64-encoded-audio>"}
```

**Server sends:**
```json
{"type": "config_ack", "config": {...}}
{"type": "status", "message": "processing"}
{"type": "audio", "data": "<base64-encoded-wav>", "sample_rate": 24000}
{"type": "error", "message": "..."}
```

## Adding Custom Speaker Voices

Place files in the `example/` directory:
1. `example/prompt_audio/<name>.wav` — Reference audio clip
2. `example/prompt_text/<name>.txt` — Transcript of the reference audio

The speaker will appear automatically in the UI and API.
