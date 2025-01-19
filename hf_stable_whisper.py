# https://github.com/jianfch/stable-ts/issues/309
from transformers import WhisperProcessor, WhisperForConditionalGeneration, GenerationConfig
import whisper
from pathlib import Path
from optimum.intel.openvino import OVModelForSpeechSeq2Seq
from transformers import pipeline
import openvino as ov
import json
import stable_whisper
import time

# Load the Whisper model
# model_id = "openai/whisper-small"
model_id = "distil-whisper/distil-small.en"
model = WhisperForConditionalGeneration.from_pretrained(model_id)
processor = WhisperProcessor.from_pretrained(model_id)
generation_config = GenerationConfig.from_pretrained(model_id)

audiofile = "how_are_you_doing_today.wav"
# audiofile = "/home/roger/sample-10min.wav"

# Load audio
audio = whisper.load_audio(audiofile, sr=16000)  # Original line
input_features = processor(audio, return_tensors="pt").input_features

# Configure OpenVINO model
ov_config = {"CACHE_DIR": ""}
model_path = Path(model_id.replace('/', '_'))

# Choose device
core = ov.Core()
options=str(core.available_devices)
print("Avaiable devices: "+options)

if not model_path.exists():
    ov_model = OVModelForSpeechSeq2Seq.from_pretrained(
        model_id, ov_config=ov_config, export=True, compile=False, load_in_8bit=False
    )
    ov_model.half()
    ov_model.save_pretrained(model_path)
else:
    ov_model = OVModelForSpeechSeq2Seq.from_pretrained(
        model_path, ov_config=ov_config, compile=False
    )

ov_model.generation_config = generation_config

device = 'GPU'  # Change this to 'GPU' if GPU is preferred
ov_model.to(device)
ov_model.compile()

# Configure pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model=ov_model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=30
)

# Initialize the model instance with pipeline
model_instance = stable_whisper.load_hf_whisper(model_id, pipeline=pipe)

start = time.time()
result = model_instance.transcribe(audio, word_timestamps=False, batch_size=20)
print(f"inference time is {time.time()-start}")

# Save result to JSON
# with open("sample.json", "w") as outfile:
#     json.dump(result, outfile)

# print(result["text"])

# mesurements with on intel-core-i5-1135g7 16GB WSL2 Ubuntu 22 
# "distil-whisper/distil-small.en" for 10 min sample (WSL2)
# this program CPU: inference time is 637.0 s
# this program GPU: inference time is 163.4/199/188 s memory 7 GB / CPU 12% / GPU 99% 3.2 GB shared GPU memory 
# 8-bit quant 130.7 sI'm 
# "openai/whisper-small" 10 min sample
# whisper.cpp with openvino GPU inference time is 178.0/214/194 s memory 2.2 GB / CPU 23% / GPU 10% (spikes only) 1.1 shared GPU memory
# this program GPU inference time is 431.0/520s memory 8.6 GB / CPU 8% GPU 99% 5GB shared GPU memory   
# "distil-whisper/distil-small.en" for 10 min sample (Windows 11)
# this program GPU: inference time is 200 s memory 7 GB / CPU 40%  base with spikes/ GPU 24% base with spikes 1.9 GB shared GPU memory 
# "distil-whisper/distil-large-v3" for 10 min sample (Windows 11)
# this program GPU: inference time is 347.4 s process 4.2 GB  / CPU 10%  base with small spikes/ GPU 80+ spikes% base with spikes 3.3 GB shared GPU memory 
# 8-bit quant 273 s

