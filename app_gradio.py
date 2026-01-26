"""
FlashLabs Chroma - Gradio Web Interface

A user-friendly web interface for the Chroma multimodal AI model.
Designed for Hugging Face Spaces deployment.
"""

import os
import tempfile

import torch
import numpy as np
import gradio as gr

# Global model variables
model = None
processor = None

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


def load_speaker_prompt(speaker_name: str):
    """Load reference audio and text for a speaker."""
    base_path = os.path.dirname(os.path.abspath(__file__))
    text_path = os.path.join(base_path, f"example/prompt_text/{speaker_name}.txt")
    audio_path = os.path.join(base_path, f"example/prompt_audio/{speaker_name}.wav")

    with open(text_path, "r", encoding="utf-8") as f:
        prompt_text = f.read()

    return [prompt_text], [audio_path]


def generate_response(audio_input, speaker, max_new_tokens, temperature):
    """Generate audio response from input audio."""
    global model, processor

    if model is None or processor is None:
        return None, "Model is still loading. Please wait..."

    if audio_input is None:
        return None, "Please upload or record an audio file."

    try:
        # audio_input is a tuple of (sample_rate, audio_data) from Gradio
        sample_rate, audio_data = audio_input

        # Save input audio to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            import soundfile as sf
            # Normalize audio data
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            sf.write(tmp_file.name, audio_data, sample_rate)
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
                    max_new_tokens=int(max_new_tokens),
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                    use_cache=True
                )

            # Decode audio
            audio_values = model.codec_model.decode(
                output.permute(0, 2, 1)
            ).audio_values

            # Convert to numpy
            audio_np = audio_values[0].cpu().detach().numpy()

            return (24000, audio_np), "Response generated successfully!"

        finally:
            # Cleanup temp file
            if os.path.exists(tmp_audio_path):
                os.unlink(tmp_audio_path)

    except Exception as e:
        return None, f"Error: {str(e)}"


# Load model at startup
print("Starting model loading...")
model, processor = load_model()

# Create Gradio interface
with gr.Blocks(
    title="FlashLabs Chroma",
    theme=gr.themes.Soft()
) as demo:
    gr.Markdown(
        """
        # FlashLabs Chroma - Voice AI

        Upload or record audio, and Chroma will respond with synthesized speech
        in your chosen voice style.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                label="Input Audio",
                type="numpy",
                sources=["upload", "microphone"]
            )

            speaker = gr.Dropdown(
                choices=AVAILABLE_SPEAKERS,
                value="scarlett_johansson",
                label="Voice Style"
            )

            with gr.Row():
                max_tokens = gr.Slider(
                    minimum=10,
                    maximum=500,
                    value=100,
                    step=10,
                    label="Max Tokens"
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=1.5,
                    value=0.7,
                    step=0.1,
                    label="Temperature"
                )

            generate_btn = gr.Button("Generate Response", variant="primary")

        with gr.Column(scale=1):
            audio_output = gr.Audio(
                label="Response Audio",
                type="numpy"
            )
            status = gr.Textbox(label="Status", interactive=False)

    generate_btn.click(
        fn=generate_response,
        inputs=[audio_input, speaker, max_tokens, temperature],
        outputs=[audio_output, status]
    )

    gr.Markdown(
        """
        ## Tips
        - **Voice Style**: Each voice has unique characteristics based on reference audio
        - **Temperature**: Higher values = more creative/varied, lower = more consistent
        - **Max Tokens**: Controls the maximum length of the response

        ## API Access
        This app also exposes a REST API. Visit `/docs` for API documentation.
        """
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False
    )
