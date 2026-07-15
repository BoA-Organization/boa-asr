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

# Limits
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB
MAX_AUDIO_DURATION_SECONDS = 40.0
TARGET_SAMPLE_RATE = 16000

# Keep-alive settings
KEEP_ALIVE_ENABLED = os.getenv("KEEP_ALIVE_ENABLED", "true").lower() == "true"
KEEP_ALIVE_INTERVAL_SECONDS = int(os.getenv("KEEP_ALIVE_INTERVAL_SECONDS", "30"))  # 30 seconds for maximum GPU activity

processor = None
model = None
asr_pipeline = None

# Prevent multiple requests from running on the same model simultaneously.
inference_lock = asyncio.Lock()

# Keep-alive task
keep_alive_task = None
shutdown_event = asyncio.Event()


def warmup_model():
    """Run a dummy inference to warm up the GPU and model."""
    print("Warming up the model...")
    
    # Create a 1-second silent audio clip
    dummy_audio = np.zeros(TARGET_SAMPLE_RATE, dtype=np.float32)
    
    with torch.inference_mode():
        asr_pipeline(
            {
                "array": dummy_audio,
                "sampling_rate": TARGET_SAMPLE_RATE,
            },
            generate_kwargs={
                "task": "transcribe",
                "language": "am",
            },
        )
    
    print("Model warmup completed.")


async def keep_alive_loop():
    """Periodically run dummy inference to keep GPU warm."""
    print(f"Keep-alive enabled: running warmup every {KEEP_ALIVE_INTERVAL_SECONDS} seconds")
    
    while not shutdown_event.is_set():
        try:
            await asyncio.sleep(KEEP_ALIVE_INTERVAL_SECONDS)
            
            if shutdown_event.is_set():
                break
            
            async with inference_lock:
                await asyncio.to_thread(warmup_model)
                
        except asyncio.CancelledError:
            print("Keep-alive task cancelled")
            break
        except Exception as error:
            print(f"Keep-alive error: {error}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor, model, asr_pipeline, keep_alive_task

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

    # Initial warmup
    warmup_model()

    # Start keep-alive task
    if KEEP_ALIVE_ENABLED:
        keep_alive_task = asyncio.create_task(keep_alive_loop())

    yield

    # Shutdown
    shutdown_event.set()
    
    if keep_alive_task:
        keep_alive_task.cancel()
        try:
            await keep_alive_task
        except asyncio.CancelledError:
            pass

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
        "keep_alive_enabled": KEEP_ALIVE_ENABLED,
        "keep_alive_interval_seconds": KEEP_ALIVE_INTERVAL_SECONDS if KEEP_ALIVE_ENABLED else None,
    }


@app.post("/warmup")
async def warmup_endpoint():
    """Manual warmup endpoint to pre-warm the GPU."""
    if asr_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="ASR model is not ready.",
        )
    
    try:
        started_at = time.perf_counter()
        
        async with inference_lock:
            await asyncio.to_thread(warmup_model)
        
        elapsed = time.perf_counter() - started_at
        
        return {
            "status": "warmup_completed",
            "warmup_time_seconds": round(elapsed, 3),
        }
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Warmup failed: {type(error).__name__}: {error}",
        ) from error


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

        # Check file size limit
        file_size = len(content)
        if file_size > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File size ({file_size / (1024 * 1024):.2f} MB) exceeds "
                    f"maximum allowed size of {MAX_FILE_SIZE_BYTES / (1024 * 1024):.0f} MB."
                ),
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

        if sample_rate != TARGET_SAMPLE_RATE:
            audio = librosa.resample(
                audio,
                orig_sr=sample_rate,
                target_sr=TARGET_SAMPLE_RATE,
            ).astype(np.float32)

        duration_seconds = len(audio) / TARGET_SAMPLE_RATE

        if duration_seconds > MAX_AUDIO_DURATION_SECONDS:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Audio duration ({duration_seconds:.2f} seconds) exceeds "
                    f"maximum allowed duration of {MAX_AUDIO_DURATION_SECONDS} seconds."
                ),
            )

        started_at = time.perf_counter()

        async with inference_lock:
            with torch.inference_mode():
                result = await asyncio.to_thread(
                    asr_pipeline,
                    {
                        "array": audio,
                        "sampling_rate": TARGET_SAMPLE_RATE,
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