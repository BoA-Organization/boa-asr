import os
import tempfile
import time

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

MODEL_ID = "b1n1yam/shook-medium-amharic-2k"

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required. No NVIDIA GPU was detected.")

device_id = int(os.getenv("GPU_ID", "0"))

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    use_safetensors=True,
    attn_implementation="sdpa",
).to(f"cuda:{device_id}")

model.eval()

asr = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch.float16,
    device=device_id,
    chunk_length_s=30,
    stride_length_s=(5, 3),
)

app = FastAPI(title="Amharic ASR GPU API")


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": MODEL_ID,
        "device": torch.cuda.get_device_name(device_id),
        "gpu_id": device_id,
    }


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"

    try:
        content = await file.read()

        with tempfile.NamedTemporaryFile(suffix=suffix) as audio_file:
            audio_file.write(content)
            audio_file.flush()

            start = time.perf_counter()

            result = asr(
                audio_file.name,
                generate_kwargs={
                    "task": "transcribe",
                    "language": "am",
                },
            )

            elapsed = time.perf_counter() - start

        return {
            "text": result["text"].strip(),
            "processing_time_seconds": round(elapsed, 3),
            "gpu": torch.cuda.get_device_name(device_id),
        }

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))