import asyncio
import io
import os
import time
from contextlib import asynccontextmanager

import librosa
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


MODEL_ID = os.getenv(
    "MODEL_ID",
    "b1n1yam/shook-medium-amharic-2k",
)

DEVICE_ID = int(os.getenv("GPU_ID", "0"))
DEVICE = f"cuda:{DEVICE_ID}"
DTYPE = torch.float16

processor = None
model = None
asr_pipeline = None

# Prevent multiple requests from running on the same model simultaneously.
inference_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor, model, asr_pipeline

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required, but PyTorch cannot access CUDA."
        )

    gpu_name = torch.cuda.get_device_name(DEVICE_ID)
    capability = torch.cuda.get_device_capability(DEVICE_ID)

    print(f"Loading model: {MODEL_ID}")
    print(f"GPU: {gpu_name}")
    print(f"GPU capability: {capability}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA build: {torch.version.cuda}")
    print(f"Supported CUDA architectures: {torch.cuda.get_arch_list()}")

    # Run a real CUDA operation before loading the model.
    try:
        test_tensor = torch.ones(1, device=DEVICE)
        print(f"CUDA tensor test: {test_tensor.item()}")
        del test_tensor
    except Exception as error:
        raise RuntimeError(
            f"PyTorch detected CUDA, but GPU computation failed: {error}"
        ) from error

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        dtype=DTYPE,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )

    model.to(DEVICE)
    model.eval()

    asr_pipeline = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=DTYPE,
        device=DEVICE_ID,
    )

    print(f"Model loaded on: {next(model.parameters()).device}")

    yield

    del asr_pipeline
    del model
    del processor

    torch.cuda.empty_cache()


app = FastAPI(
    title="Amharic ASR GPU API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "service": "Amharic ASR GPU API",
        "docs": "/docs",
        "health": "/health",
        "transcribe": "/transcribe",
    }


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="ASR model is not loaded.",
        )

    return {
        "status": "healthy",
        "model": MODEL_ID,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(DEVICE_ID),
        "gpu_capability": torch.cuda.get_device_capability(DEVICE_ID),
        "pytorch_version": torch.__version__,
        "pytorch_cuda": torch.version.cuda,
        "supported_architectures": torch.cuda.get_arch_list(),
        "device": str(next(model.parameters()).device),
    }


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if asr_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="ASR model is not ready.",
        )

    try:
        content = await file.read()

        if not content:
            raise HTTPException(
                status_code=400,
                detail="The uploaded audio file is empty.",
            )

        try:
            audio, sample_rate = sf.read(
                io.BytesIO(content),
                dtype="float32",
                always_2d=False,
            )
        except Exception as error:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Unable to decode the audio file. "
                    "Use WAV, FLAC, or OGG audio. "
                    f"Decoder error: {error}"
                ),
            ) from error

        if audio.size == 0:
            raise HTTPException(
                status_code=400,
                detail="The decoded audio contains no samples.",
            )

        # Convert stereo or multichannel audio to mono.
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Remove invalid values.
        audio = np.nan_to_num(
            audio,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)

        if sample_rate != 16000:
            audio = librosa.resample(
                audio,
                orig_sr=sample_rate,
                target_sr=16000,
            ).astype(np.float32)

        duration_seconds = len(audio) / 16000

        if duration_seconds < 0.1:
            raise HTTPException(
                status_code=400,
                detail="The audio is too short to transcribe.",
            )

        started_at = time.perf_counter()

        async with inference_lock:
            with torch.inference_mode():
                result = await asyncio.to_thread(
                    asr_pipeline,
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
            "text": result.get("text", "").strip(),
            "audio_duration_seconds": round(duration_seconds, 3),
            "processing_time_seconds": round(elapsed, 3),
            "real_time_factor": round(
                elapsed / duration_seconds,
                3,
            ),
            "model": MODEL_ID,
            "gpu": torch.cuda.get_device_name(DEVICE_ID),
            "device": DEVICE,
        }

    except HTTPException:
        raise

    except torch.cuda.OutOfMemoryError as error:
        torch.cuda.empty_cache()

        raise HTTPException(
            status_code=503,
            detail="The GPU ran out of memory while processing the audio.",
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {type(error).__name__}: {error}",
        ) from error

    finally:
        await file.close()