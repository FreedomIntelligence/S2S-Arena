import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os
import json


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)


def process_directory(directory):
    results = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.wav', '.mp3')):
                file_path = os.path.join(root, file)

                transcription = pipe(file_path)["text"]

                results.append({
                    'filename': os.path.relpath(file_path, start=directory),
                    'transcription': transcription
                })

    json_path = os.path.join(directory, 'transcriptions.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def list_subdirectories(base_path):
    entries = os.listdir(base_path)
    return [os.path.join(base_path, entry) for entry in entries if os.path.isdir(os.path.join(base_path, entry))]



if __name__ == '__main__':
    base_path = 'PathToYourDataset/input'

    for dir_path in list_subdirectories(base_path):
        process_directory(dir_path)

    pass
