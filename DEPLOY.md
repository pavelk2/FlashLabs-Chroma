# Deploying Chroma as a WebSocket Voice Server

This guide covers deploying the FlashLabs Chroma speech-to-speech model as a WebSocket server, both locally and on [Modal](https://modal.com).

## Architecture

```
Client (audio) ──WebSocket──> FastAPI Server ──> Chroma Model ──> Audio Response
```

The server accepts audio input via WebSocket, processes it through the Chroma model, and returns generated speech audio. It supports voice cloning via reference audio presets or custom uploads.

## Files

| File | Purpose |
|------|---------|
| `server.py` | FastAPI WebSocket server with Chroma integration |
| `modal_app.py` | Modal GPU deployment configuration |
| `requirements-server.txt` | Server dependencies |

## Local Setup (GPU Required)

### Prerequisites
- Python 3.11+
- CUDA 12.6+ compatible GPU with 16GB+ VRAM
- ~10GB disk space for model weights

### Install and Run

```bash
# Install dependencies
pip install -r requirements-server.txt

# Run the server (downloads model on first run)
python server.py --model-id FlashLabs/Chroma-4B --port 8000

# Or use a local model path
python server.py --model-id /path/to/Chroma-4B --port 8000
```

The server starts at `http://localhost:8000`. The WebSocket endpoint is at `ws://localhost:8000/ws/chat`.

## Modal Deployment

### Prerequisites

```bash
pip install modal
modal setup  # Authenticate with Modal
```

### Deploy

```bash
# Step 1: Download model weights to Modal volume (first time only)
modal run modal_app.py

# Step 2: Deploy the server
modal deploy modal_app.py
```

Modal will print the server URL after deployment. The WebSocket endpoint will be at `wss://<your-app>.modal.run/ws/chat`.

### Configuration

Edit `modal_app.py` to adjust:
- **GPU type**: Change `modal.gpu.A100(size="40GB")` to `modal.gpu.A10G()` for lower cost
- **Idle timeout**: `container_idle_timeout=300` (seconds before spinning down)
- **Concurrency**: `allow_concurrent_inputs=1` (one request at a time per container)

## WebSocket API

### Connect

```
ws://localhost:8000/ws/chat       # Local
wss://<app>.modal.run/ws/chat     # Modal
```

### Message Types

#### Configure Session

Send a config message to set voice, system prompt, or generation parameters:

```json
{
  "type": "config",
  "system_prompt": "You are a helpful assistant.",
  "voice": "scarlett_johansson",
  "max_new_tokens": 500,
  "temperature": 0.7,
  "top_p": 0.9
}
```

Available voice presets: `ariana_grande`, `donald_trump`, `lebron_james`, `scarlett_johansson`

#### Send Audio (JSON)

```json
{
  "type": "audio",
  "data": "<base64-encoded WAV or PCM audio>"
}
```

#### Send Audio (Binary)

Send raw binary audio bytes directly as a WebSocket binary frame. Accepts WAV format or raw PCM (16-bit signed, mono, 24kHz).

#### Upload Custom Voice Reference

```json
{
  "type": "voice_ref",
  "audio": "<base64-encoded reference audio>",
  "text": "Optional transcription of the reference audio"
}
```

### Server Responses

#### Audio Response

```json
{
  "type": "audio",
  "data": "<base64-encoded WAV>",
  "sample_rate": 24000,
  "format": "wav"
}
```

#### Status Updates

```json
{"type": "status", "message": "processing"}
{"type": "status", "message": "done"}
```

#### Errors

```json
{"type": "error", "message": "description of error"}
```

## HTTP Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check, model status, available voices |
| `/voices` | GET | List available voice presets |

## Client Example (Python)

```python
import asyncio
import base64
import json
import websockets

async def chat(audio_file: str, server_url: str = "ws://localhost:8000/ws/chat"):
    async with websockets.connect(server_url) as ws:
        # Optional: configure voice
        await ws.send(json.dumps({
            "type": "config",
            "voice": "scarlett_johansson"
        }))
        config_ack = await ws.recv()
        print("Config:", json.loads(config_ack))

        # Send audio
        with open(audio_file, "rb") as f:
            audio_bytes = f.read()

        await ws.send(json.dumps({
            "type": "audio",
            "data": base64.b64encode(audio_bytes).decode()
        }))

        # Receive responses
        while True:
            resp = json.loads(await ws.recv())
            if resp["type"] == "audio":
                wav_data = base64.b64decode(resp["data"])
                with open("response.wav", "wb") as f:
                    f.write(wav_data)
                print(f"Saved response audio ({len(wav_data)} bytes)")
            elif resp["type"] == "status" and resp["message"] == "done":
                break
            else:
                print("Server:", resp)

asyncio.run(chat("example/make_taco.wav"))
```

## Notes

- Audio output is always 24kHz mono WAV (PCM 16-bit)
- Input audio is automatically resampled to 24kHz if needed
- The model requires `trust_remote_code=True` for loading
- First request after cold start may be slower due to CUDA kernel compilation
- Modal's memory snapshot feature speeds up subsequent cold starts
