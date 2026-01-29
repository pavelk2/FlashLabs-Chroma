"""
Modal deployment for the Chroma WebSocket voice server.

Prerequisites:
    1. Accept model access at https://huggingface.co/FlashLabs/Chroma-4B
    2. Create a HuggingFace token at https://huggingface.co/settings/tokens
    3. Add it as a Modal secret:
       modal secret create huggingface HF_TOKEN=hf_your_token_here

Deploy with:
    modal deploy modal_app.py

Run locally (dev mode):
    modal serve modal_app.py
"""

from pathlib import Path

import modal

LOCAL_DIR = Path(__file__).parent

MODELS_DIR = "/models"
MODEL_ID = "FlashLabs/Chroma-4B"

# --------------- HuggingFace secret for gated model access ---------------

hf_secret = modal.Secret.from_name("huggingface")

# --------------- Image definition ---------------
# Model weights are downloaded at image build time, so HuggingFace is only
# contacted once during `modal deploy`, never at runtime.


def download_model():
    """Called during image build to download weights into the image layer."""
    import os
    from huggingface_hub import snapshot_download

    snapshot_download(
        MODEL_ID,
        local_dir=f"{MODELS_DIR}/Chroma-4B",
        token=os.environ["HF_TOKEN"],
    )


chroma_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # Core ML
        "torch==2.7.1",
        "torchaudio==2.7.1",
        "transformers==5.0.0rc0",
        "accelerate>=1.7.0",
        "safetensors>=0.5.0",
        "huggingface-hub>=1.3.0",
        # Audio
        "av>=14.0.0",
        "librosa>=0.11.0",
        "audioread>=3.0.0",
        "soundfile>=0.13.0",
        # Server
        "fastapi[standard]",
        "uvicorn[standard]",
        "websockets",
        # Other
        "numpy>=2.2.0",
        "pillow>=11.0.0",
    )
    .run_commands(
        "apt-get update && apt-get install -y libsndfile1 ffmpeg && rm -rf /var/lib/apt/lists/*"
    )
    .run_function(download_model, secrets=[hf_secret])
    .add_local_file(str(LOCAL_DIR / "server.py"), "/root/server.py")
    .add_local_dir(str(LOCAL_DIR / "static"), "/root/static")
    .add_local_dir(str(LOCAL_DIR / "example"), "/root/example")
    .add_local_dir(str(LOCAL_DIR / "chroma"), "/root/chroma")
)


# --------------- Modal App ---------------

app = modal.App("chroma-voice-server", image=chroma_image)


@app.cls(
    gpu="A100-40GB",
    scaledown_window=300,
    timeout=600,
    enable_memory_snapshot=True,
)
@modal.concurrent(max_inputs=1)
class ChromaServer:
    """Modal class that hosts the Chroma WebSocket server."""

    @modal.enter(snap=True)
    def load_model(self):
        """Load the model at container start (snapshotted for fast cold starts)."""
        import server as srv

        srv.load_model(f"{MODELS_DIR}/Chroma-4B")
        self.app = srv.app

    @modal.enter(snap=False)
    def setup_post_snapshot(self):
        """Re-initialize anything that can't be snapshotted (e.g. CUDA state)."""
        pass

    @modal.asgi_app()
    def serve(self):
        return self.app
