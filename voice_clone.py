import torch
from TTS.api import TTS
import os

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Example voice cloning

## XTTS-V2
### tts_models/multilingual/multi-dataset/xtts_v2
## BARK
### tts_models/multilingual/multi-dataset/bark
## YOUR-TTS
### tts_models/multilingual/multi-dataset/your_tts

tts = TTS(model_name="tts_models/multilingual/multi-dataset/bark").to(device)

input_text = "최민식입니다. 안녕하세요. 만나서 반가워요."

ref_wav_path = "samples_ref/sample_ref_01.wav"

output_wav_path = "sample_cloned_05_bark.wav"

tts.tts_to_file(input_text, speaker_wav=ref_wav_path, language="ko", file_path=output_wav_path)