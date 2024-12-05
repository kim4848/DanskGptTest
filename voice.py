import time
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import soundfile as sf
import intel_extension_for_pytorch as ipex
import torch_directml


# Check for GPU or CPU
device = device = torch_directml.device()
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Model and processor
model_id = "syvai/hviske-v2"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
if device == "cpu":
    model = ipex.optimize(model.eval())
model.to(device)

# Reset forced_decoder_ids to None
model.config.forced_decoder_ids = None

processor = AutoProcessor.from_pretrained(model_id)

# Pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# Load your own WAV file
wav_file_path = "optagelse.wav"  # Replace with the path to your WAV file
audio, sample_rate = sf.read(wav_file_path)

# Convert to the format expected by the pipeline
if sample_rate != processor.feature_extractor.sampling_rate:
    raise ValueError(
        f"Audio file must be in {processor.feature_extractor.sampling_rate} Hz. Currently, it is {sample_rate} Hz."
    )

start = time.time()
temp = "cuda" if torch.cuda.is_available() else "cpu"
print(temp)

# Use the pipeline for speech recognition
result = pipe(audio, generate_kwargs={
    "max_new_tokens": 200,
    "return_timestamps": True,
    "language": "danish"
})
print(result["text"])

# Print time it took to run the code in minutes
print("Time it took to run the code in minutes rounded to 2 decimals")
print(round((time.time() - start) / 60, 2))

# Print time it took to run the code in seconds
print("Time it took to run the code in seconds rounded to 2 decimals")
print(round(time.time() - start, 2))
