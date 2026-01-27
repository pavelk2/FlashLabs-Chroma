"""
FlashLabs Chroma Voice Agent Web Service

A FastAPI-based web service that exposes the Chroma speech-to-speech model
as a real-time voice agent, similar to https://www.flashlabs.ai/flashai-voice-agents
"""

import os
import io
import uuid
import time
import base64
import logging
import asyncio
import tempfile
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import torch
import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global model references (populated at startup)
# ---------------------------------------------------------------------------
model = None
processor = None

SPEAKERS_DIR = Path(__file__).parent / "example"
MODEL_ID = os.environ.get("CHROMA_MODEL_ID", "FlashLabs/Chroma-4B")
DEVICE = os.environ.get("CHROMA_DEVICE", "auto")
SAMPLE_RATE_OUT = 24_000  # Chroma outputs 24 kHz audio


def _load_model():
    """Load the Chroma model and processor."""
    from transformers import AutoModelForCausalLM, AutoProcessor

    logger.info("Loading Chroma model from %s ...", MODEL_ID)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        device_map=DEVICE,
    )
    _processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    logger.info("Model loaded successfully on %s", _model.device)
    return _model, _processor


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor
    model, processor = await asyncio.to_thread(_load_model)
    yield


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="FlashLabs Chroma Voice Agent",
    description="Real-time voice agent powered by the Chroma speech-to-speech model",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def get_available_speakers() -> list[dict]:
    """Scan the example directory for available speaker voices."""
    speakers = []
    prompt_audio_dir = SPEAKERS_DIR / "prompt_audio"
    prompt_text_dir = SPEAKERS_DIR / "prompt_text"

    if not prompt_audio_dir.exists():
        return speakers

    for audio_file in sorted(prompt_audio_dir.glob("*.wav")):
        name = audio_file.stem
        text_file = prompt_text_dir / f"{name}.txt"
        if text_file.exists():
            speakers.append({
                "id": name,
                "name": name.replace("_", " ").title(),
                "audio_path": str(audio_file),
                "text_path": str(text_file),
            })
    return speakers


def load_speaker(speaker_id: str):
    """Load prompt text and audio path for a given speaker."""
    text_path = SPEAKERS_DIR / "prompt_text" / f"{speaker_id}.txt"
    audio_path = SPEAKERS_DIR / "prompt_audio" / f"{speaker_id}.wav"

    if not text_path.exists() or not audio_path.exists():
        raise FileNotFoundError(f"Speaker '{speaker_id}' not found")

    with open(text_path, "r", encoding="utf-8") as f:
        prompt_text = f.read()

    return [prompt_text], [str(audio_path)]


SYSTEM_PROMPT = (
    "You are Chroma, an advanced virtual human created by the FlashLabs. "
    "You possess the ability to understand auditory inputs and generate both text and speech."
)


@torch.inference_mode()
def run_inference(
    audio_bytes: bytes,
    speaker_id: str = "scarlett_johansson",
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
) -> np.ndarray:
    """
    Run the Chroma model: audio in -> audio out.

    Returns a numpy array of float32 samples at 24 kHz.
    """
    # Save input audio to a temporary file (processor expects a file path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        prompt_text, prompt_audio = load_speaker(speaker_id)

        conversation = [[
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [{"type": "audio", "audio": tmp_path}],
            },
        ]]

        inputs = processor(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
            prompt_audio=prompt_audio,
            prompt_text=prompt_text,
        )

        device = model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            use_cache=True,
            output_audio=True,
        )

        if output.audio and len(output.audio) > 0:
            audio_np = output.audio[0].cpu().float().numpy().squeeze()
        else:
            audio_np = np.zeros(SAMPLE_RATE_OUT, dtype=np.float32)

        return audio_np
    finally:
        os.unlink(tmp_path)


def numpy_to_wav_bytes(audio_np: np.ndarray, sample_rate: int = SAMPLE_RATE_OUT) -> bytes:
    """Convert a numpy audio array to WAV bytes."""
    buf = io.BytesIO()
    sf.write(buf, audio_np, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# REST API endpoints
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    """Serve the main voice agent UI."""
    index_path = Path(__file__).parent / "static" / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>FlashLabs Chroma Voice Agent</h1><p>Static files not found.</p>")


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_id": MODEL_ID,
        "device": str(model.device) if model else None,
    }


@app.get("/api/speakers")
async def list_speakers():
    """List available speaker voices for cloning."""
    return {"speakers": get_available_speakers()}


@app.post("/api/generate")
async def generate_audio(
    audio: UploadFile = File(..., description="Input audio file (WAV/MP3/etc.)"),
    speaker: str = Form("scarlett_johansson", description="Speaker voice ID"),
    max_new_tokens: int = Form(200, description="Maximum audio tokens to generate"),
    temperature: float = Form(0.7, description="Sampling temperature"),
    top_k: int = Form(50, description="Top-k sampling"),
    top_p: float = Form(0.9, description="Nucleus sampling threshold"),
):
    """
    Generate a voice response from an audio input.

    Send an audio file and receive a WAV audio response in the selected speaker voice.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    try:
        audio_np = await asyncio.to_thread(
            run_inference,
            audio_bytes,
            speaker,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    wav_bytes = numpy_to_wav_bytes(audio_np)
    return StreamingResponse(
        io.BytesIO(wav_bytes),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=response.wav"},
    )


# ---------------------------------------------------------------------------
# WebSocket endpoint for real-time voice interaction
# ---------------------------------------------------------------------------
@app.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time voice interaction.

    Protocol (JSON messages):
    -> Client sends: {"type": "config", "speaker": "...", "max_new_tokens": 200, ...}
    -> Client sends: {"type": "audio", "data": "<base64-wav-bytes>"}
    <- Server sends: {"type": "audio", "data": "<base64-wav-bytes>", "sample_rate": 24000}
    <- Server sends: {"type": "error", "message": "..."}
    """
    await websocket.accept()
    logger.info("WebSocket client connected")

    # Per-connection config
    config = {
        "speaker": "scarlett_johansson",
        "max_new_tokens": 200,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9,
    }

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "")

            if msg_type == "config":
                config.update({
                    k: data[k] for k in config if k in data
                })
                await websocket.send_json({"type": "config_ack", "config": config})

            elif msg_type == "audio":
                if model is None:
                    await websocket.send_json({"type": "error", "message": "Model not loaded"})
                    continue

                audio_b64 = data.get("data", "")
                if not audio_b64:
                    await websocket.send_json({"type": "error", "message": "No audio data"})
                    continue

                try:
                    audio_bytes = base64.b64decode(audio_b64)
                    await websocket.send_json({"type": "status", "message": "processing"})

                    audio_np = await asyncio.to_thread(
                        run_inference,
                        audio_bytes,
                        config["speaker"],
                        config["max_new_tokens"],
                        config["temperature"],
                        config["top_p"],
                        config["top_k"],
                    )

                    wav_bytes = numpy_to_wav_bytes(audio_np)
                    resp_b64 = base64.b64encode(wav_bytes).decode("utf-8")

                    await websocket.send_json({
                        "type": "audio",
                        "data": resp_b64,
                        "sample_rate": SAMPLE_RATE_OUT,
                    })
                except Exception as e:
                    logger.exception("WebSocket inference error")
                    await websocket.send_json({"type": "error", "message": str(e)})

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.exception("WebSocket error")


# ---------------------------------------------------------------------------
# Mount static files (must be last so it doesn't override API routes)
# ---------------------------------------------------------------------------
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("server:app", host=host, port=port, workers=1)
