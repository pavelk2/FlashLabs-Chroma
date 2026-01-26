"""
FlashLabs Chroma Web Service

A FastAPI-based web service for the Chroma multimodal AI model.
Supports audio input/output chat with voice cloning capabilities.
"""

import os
import io
import base64
import tempfile
from typing import Optional, List
from contextlib import asynccontextmanager

import torch
import soundfile as sf
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Global model variables
model = None
processor = None

# Available speakers for voice cloning
AVAILABLE_SPEAKERS = [
    "scarlett_johansson",
    "donald_trump",
    "ariana_grande",
    "lebron_james"
]

SYSTEM_PROMPT = (
    "You are Chroma, an advanced virtual human created by the FlashLabs. "
    "You possess the ability to understand auditory inputs and generate both text and speech."
)


def load_model():
    """Load the Chroma model and processor."""
    from transformers import AutoModelForCausalLM, AutoProcessor

    model_id = os.environ.get("MODEL_ID", "FlashLabs/Chroma-4B")

    print(f"Loading model from {model_id}...")

    _model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    _processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True
    )

    print("Model loaded successfully!")
    return _model, _processor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - load model on startup."""
    global model, processor
    model, processor = load_model()
    yield
    # Cleanup on shutdown
    del model, processor
    torch.cuda.empty_cache()


app = FastAPI(
    title="FlashLabs Chroma API",
    description="Multimodal AI model for spoken dialogue with voice cloning",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    """Request body for chat endpoint with base64 audio."""
    audio_base64: str
    speaker: Optional[str] = "scarlett_johansson"
    max_new_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9


class ChatResponse(BaseModel):
    """Response from chat endpoint."""
    audio_base64: str
    sample_rate: int = 24000


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str
    cuda_available: bool


class SpeakersResponse(BaseModel):
    """Available speakers response."""
    speakers: List[str]


def load_speaker_prompt(speaker_name: str):
    """Load reference audio and text for a speaker."""
    base_path = os.path.dirname(os.path.abspath(__file__))
    text_path = os.path.join(base_path, f"example/prompt_text/{speaker_name}.txt")
    audio_path = os.path.join(base_path, f"example/prompt_audio/{speaker_name}.wav")

    if not os.path.exists(text_path) or not os.path.exists(audio_path):
        raise HTTPException(
            status_code=400,
            detail=f"Speaker '{speaker_name}' not found. Available: {AVAILABLE_SPEAKERS}"
        )

    with open(text_path, "r", encoding="utf-8") as f:
        prompt_text = f.read()

    return [prompt_text], [audio_path]


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the service is healthy and model is loaded."""
    return HealthResponse(
        status="healthy" if model is not None else "loading",
        model_loaded=model is not None,
        device=str(model.device) if model is not None else "not loaded",
        cuda_available=torch.cuda.is_available()
    )


@app.get("/speakers", response_model=SpeakersResponse)
async def list_speakers():
    """List available speakers for voice cloning."""
    return SpeakersResponse(speakers=AVAILABLE_SPEAKERS)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process audio input and generate speech response.

    - **audio_base64**: Base64-encoded audio file (WAV format recommended)
    - **speaker**: Voice style to use (default: scarlett_johansson)
    - **max_new_tokens**: Maximum tokens to generate (default: 100)
    - **temperature**: Sampling temperature (default: 0.7)
    - **top_p**: Top-p sampling parameter (default: 0.9)
    """
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if request.speaker not in AVAILABLE_SPEAKERS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid speaker. Available: {AVAILABLE_SPEAKERS}"
        )

    try:
        # Decode input audio
        audio_bytes = base64.b64decode(request.audio_base64)

        # Save to temporary file for processing
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_audio_path = tmp_file.name

        try:
            # Build conversation
            conversation = [[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [{"type": "audio", "audio": tmp_audio_path}],
                },
            ]]

            # Load speaker prompt
            prompt_text, prompt_audio = load_speaker_prompt(request.speaker)

            # Process inputs
            inputs = processor(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
                prompt_audio=prompt_audio,
                prompt_text=prompt_text
            )

            # Move to device
            device = model.device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=request.max_new_tokens,
                    do_sample=True,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    use_cache=True
                )

            # Decode audio
            audio_values = model.codec_model.decode(
                output.permute(0, 2, 1)
            ).audio_values

            # Convert to numpy and encode as base64
            audio_np = audio_values[0].cpu().detach().numpy()

            # Write to bytes buffer
            buffer = io.BytesIO()
            sf.write(buffer, audio_np, 24000, format="WAV")
            buffer.seek(0)

            audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")

            return ChatResponse(audio_base64=audio_base64, sample_rate=24000)

        finally:
            # Cleanup temp file
            if os.path.exists(tmp_audio_path):
                os.unlink(tmp_audio_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/upload")
async def chat_upload(
    audio: UploadFile = File(...),
    speaker: str = Form(default="scarlett_johansson"),
    max_new_tokens: int = Form(default=100),
    temperature: float = Form(default=0.7),
    top_p: float = Form(default=0.9),
):
    """
    Process uploaded audio file and return audio response.

    This endpoint accepts multipart form data for easier testing.
    Returns the audio file directly as a WAV response.
    """
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if speaker not in AVAILABLE_SPEAKERS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid speaker. Available: {AVAILABLE_SPEAKERS}"
        )

    try:
        # Read uploaded audio
        audio_bytes = await audio.read()

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_audio_path = tmp_file.name

        try:
            # Build conversation
            conversation = [[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [{"type": "audio", "audio": tmp_audio_path}],
                },
            ]]

            # Load speaker prompt
            prompt_text, prompt_audio = load_speaker_prompt(speaker)

            # Process inputs
            inputs = processor(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
                prompt_audio=prompt_audio,
                prompt_text=prompt_text
            )

            # Move to device
            device = model.device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    use_cache=True
                )

            # Decode audio
            audio_values = model.codec_model.decode(
                output.permute(0, 2, 1)
            ).audio_values

            # Convert to numpy
            audio_np = audio_values[0].cpu().detach().numpy()

            # Write to bytes buffer
            buffer = io.BytesIO()
            sf.write(buffer, audio_np, 24000, format="WAV")
            buffer.seek(0)

            return Response(
                content=buffer.read(),
                media_type="audio/wav",
                headers={"Content-Disposition": "attachment; filename=response.wav"}
            )

        finally:
            # Cleanup temp file
            if os.path.exists(tmp_audio_path):
                os.unlink(tmp_audio_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=False
    )
