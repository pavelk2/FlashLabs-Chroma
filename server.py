"""
FastAPI WebSocket server for FlashLabs Chroma speech-to-speech model.

Provides real-time voice conversation via WebSocket, with support for
voice cloning through reference audio and configurable system prompts.
"""

import base64
import io
import json
import logging
import struct
import tempfile
import os
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger("chroma-server")
logging.basicConfig(level=logging.INFO)

SAMPLE_RATE = 24000
PROMPT_AUDIO_DIR = Path(__file__).parent / "example" / "prompt_audio"
PROMPT_TEXT_DIR = Path(__file__).parent / "example" / "prompt_text"

DEFAULT_SYSTEM_PROMPT = (
    "You are Chroma, an advanced virtual human created by the FlashLabs. "
    "You possess the ability to understand auditory inputs and generate both text and speech."
)

AVAILABLE_VOICES = {
    name.stem: name.stem
    for name in PROMPT_AUDIO_DIR.glob("*.wav")
} if PROMPT_AUDIO_DIR.exists() else {}

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Chroma Voice Server")

# Global model references (loaded once at startup)
model = None
processor = None


def load_model(model_id: str = "FlashLabs/Chroma-4B"):
    """Load the Chroma model and processor."""
    global model, processor
    from transformers import AutoModelForCausalLM, AutoProcessor

    logger.info("Loading Chroma model from %s...", model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    logger.info("Model loaded successfully on %s", model.device)


def load_voice_prompt(speaker_name: str):
    """Load prompt audio and text for voice cloning."""
    text_path = PROMPT_TEXT_DIR / f"{speaker_name}.txt"
    audio_path = PROMPT_AUDIO_DIR / f"{speaker_name}.wav"

    if not audio_path.exists():
        raise FileNotFoundError(f"Voice preset '{speaker_name}' not found")

    prompt_text = ""
    if text_path.exists():
        prompt_text = text_path.read_text(encoding="utf-8")

    return [prompt_text], [str(audio_path)]


def audio_bytes_to_file(audio_bytes: bytes) -> str:
    """
    Convert raw audio bytes to a temporary WAV file path.
    Accepts either WAV or raw PCM (16-bit signed, mono, 24kHz).
    Returns the path to a temporary WAV file.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        # Try to read as an existing audio format (WAV, FLAC, etc.)
        buf = io.BytesIO(audio_bytes)
        data, sr = sf.read(buf)
        if data.ndim > 1:
            data = data.mean(axis=1)
        # Resample to 24kHz if needed
        if sr != SAMPLE_RATE:
            tensor = torch.from_numpy(data).float().unsqueeze(0)
            tensor = torchaudio.functional.resample(tensor, sr, SAMPLE_RATE)
            data = tensor.squeeze(0).numpy()
        sf.write(tmp.name, data, SAMPLE_RATE)
    except Exception:
        # Assume raw PCM: signed 16-bit little-endian, mono, 24kHz
        num_samples = len(audio_bytes) // 2
        samples = struct.unpack(f"<{num_samples}h", audio_bytes[: num_samples * 2])
        data = np.array(samples, dtype=np.float32) / 32768.0
        sf.write(tmp.name, data, SAMPLE_RATE)

    return tmp.name


def generate_speech(
    audio_path: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    voice_name: Optional[str] = None,
    custom_prompt_audio: Optional[str] = None,
    custom_prompt_text: Optional[str] = None,
    max_new_tokens: int = 500,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> tuple[np.ndarray, int]:
    """
    Run Chroma inference on an audio input.
    Returns (audio_array, sample_rate).
    """
    conversation = [[
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [{"type": "audio", "audio": audio_path}],
        },
    ]]

    # Resolve voice prompt
    prompt_text = [""]
    prompt_audio = [str(PROMPT_AUDIO_DIR / "scarlett_johansson.wav")] if PROMPT_AUDIO_DIR.exists() else [""]

    if voice_name and voice_name in AVAILABLE_VOICES:
        prompt_text, prompt_audio = load_voice_prompt(voice_name)
    elif custom_prompt_audio:
        prompt_audio = [custom_prompt_audio]
        prompt_text = [custom_prompt_text or ""]

    inputs = processor(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
        prompt_audio=prompt_audio,
        prompt_text=prompt_text,
    )

    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            use_cache=True,
        )

    audio_values = model.codec_model.decode(output.permute(0, 2, 1)).audio_values
    audio_np = audio_values[0].cpu().detach().numpy().squeeze()

    return audio_np, SAMPLE_RATE


class SessionState:
    """Per-connection session state."""

    def __init__(self):
        self.system_prompt: str = DEFAULT_SYSTEM_PROMPT
        self.voice_name: Optional[str] = None
        self.custom_prompt_audio: Optional[str] = None
        self.custom_prompt_text: Optional[str] = None
        self.max_new_tokens: int = 500
        self.temperature: float = 0.7
        self.top_p: float = 0.9


# --------------- HTTP Endpoints ---------------

@app.get("/health")
async def health():
    return JSONResponse({
        "status": "ok",
        "model_loaded": model is not None,
        "available_voices": list(AVAILABLE_VOICES.keys()),
    })


@app.get("/voices")
async def list_voices():
    return JSONResponse({
        "voices": list(AVAILABLE_VOICES.keys()),
    })


# --------------- WebSocket Endpoint ---------------

@app.websocket("/ws/chat")
async def websocket_chat(ws: WebSocket):
    await ws.accept()
    session = SessionState()
    logger.info("WebSocket client connected")

    try:
        while True:
            raw = await ws.receive()

            # Handle binary audio frames directly
            if "bytes" in raw and raw["bytes"]:
                audio_bytes = raw["bytes"]
                await _handle_audio(ws, session, audio_bytes)
                continue

            # Handle text (JSON) messages
            if "text" in raw and raw["text"]:
                try:
                    msg = json.loads(raw["text"])
                except json.JSONDecodeError:
                    await ws.send_json({"type": "error", "message": "Invalid JSON"})
                    continue

                msg_type = msg.get("type", "")

                if msg_type == "config":
                    _apply_config(session, msg)
                    await ws.send_json({"type": "config_ack", "message": "Configuration updated"})

                elif msg_type == "audio":
                    audio_data = msg.get("data")
                    if not audio_data:
                        await ws.send_json({"type": "error", "message": "Missing audio data"})
                        continue
                    audio_bytes = base64.b64decode(audio_data)
                    await _handle_audio(ws, session, audio_bytes)

                elif msg_type == "voice_ref":
                    # Upload custom reference audio for voice cloning
                    audio_data = msg.get("audio")
                    text_data = msg.get("text", "")
                    if not audio_data:
                        await ws.send_json({"type": "error", "message": "Missing reference audio"})
                        continue
                    ref_bytes = base64.b64decode(audio_data)
                    ref_path = audio_bytes_to_file(ref_bytes)
                    session.custom_prompt_audio = ref_path
                    session.custom_prompt_text = text_data
                    session.voice_name = None  # Custom overrides preset
                    await ws.send_json({"type": "voice_ref_ack", "message": "Voice reference set"})

                else:
                    await ws.send_json({"type": "error", "message": f"Unknown message type: {msg_type}"})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.exception("WebSocket error")
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


def _apply_config(session: SessionState, msg: dict):
    """Apply configuration changes from a config message."""
    if "system_prompt" in msg:
        session.system_prompt = msg["system_prompt"]
    if "voice" in msg:
        session.voice_name = msg["voice"]
        session.custom_prompt_audio = None
        session.custom_prompt_text = None
    if "max_new_tokens" in msg:
        session.max_new_tokens = min(int(msg["max_new_tokens"]), 2000)
    if "temperature" in msg:
        session.temperature = float(msg["temperature"])
    if "top_p" in msg:
        session.top_p = float(msg["top_p"])


async def _handle_audio(ws: WebSocket, session: SessionState, audio_bytes: bytes):
    """Process incoming audio and send back generated speech."""
    tmp_path = None
    try:
        await ws.send_json({"type": "status", "message": "processing"})

        tmp_path = audio_bytes_to_file(audio_bytes)
        audio_np, sr = generate_speech(
            audio_path=tmp_path,
            system_prompt=session.system_prompt,
            voice_name=session.voice_name,
            custom_prompt_audio=session.custom_prompt_audio,
            custom_prompt_text=session.custom_prompt_text,
            max_new_tokens=session.max_new_tokens,
            temperature=session.temperature,
            top_p=session.top_p,
        )

        # Encode response audio as WAV
        buf = io.BytesIO()
        sf.write(buf, audio_np, sr, format="WAV", subtype="PCM_16")
        wav_bytes = buf.getvalue()

        # Send as base64 JSON
        await ws.send_json({
            "type": "audio",
            "data": base64.b64encode(wav_bytes).decode("ascii"),
            "sample_rate": sr,
            "format": "wav",
        })

        await ws.send_json({"type": "status", "message": "done"})

    except Exception as e:
        logger.exception("Error processing audio")
        await ws.send_json({"type": "error", "message": str(e)})
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# --------------- Static files (web UI) ---------------

if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


# --------------- Local entrypoint ---------------

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Chroma Voice WebSocket Server")
    parser.add_argument("--model-id", default="FlashLabs/Chroma-4B", help="HuggingFace model ID or local path")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    load_model(args.model_id)
    uvicorn.run(app, host=args.host, port=args.port)
