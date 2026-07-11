import io
import os
import time

import librosa
import soundfile as sf
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


MODEL_ID = os.getenv(
    "MODEL_ID",
    "b1n1yam/shook-medium-amharic-2k",
)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required, but no CUDA GPU was detected.")

device_id = 0
dtype = torch.float16

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_ID,
    dtype=dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
).to(f"cuda:{device_id}")

model.eval()

asr = pipeline(
    task="automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    dtype=dtype,
    device=device_id,
)

app = FastAPI(title="Amharic ASR GPU API")


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": MODEL_ID,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(device_id),
        "device": str(next(model.parameters()).device),
    }


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        content = await file.read()

        if not content:
            raise HTTPException(
                status_code=400,
                detail="The uploaded audio file is empty.",
            )

        audio, sample_rate = sf.read(
            io.BytesIO(content),
            dtype="float32",
        )

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sample_rate != 16000:
            audio = librosa.resample(
                audio,
                orig_sr=sample_rate,
                target_sr=16000,
            )

        started_at = time.perf_counter()

        result = asr(
            {
                "array": audio,
                "sampling_rate": 16000,
            },
            generate_kwargs={
                "task": "transcribe",
                "language": "am",
            },
        )

        elapsed = time.perf_counter() - started_at

        return {
            "text": result["text"].strip(),
            "processing_time_seconds": round(elapsed, 3),
            "gpu": torch.cuda.get_device_name(device_id),
        }

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {error}",
        ) from error