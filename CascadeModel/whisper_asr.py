import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os
import json


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "/mnt/user/bufan/model_cache/whisper-large-v3"

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
    # 存储结果的列表
    results = []

    # 遍历目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.wav', '.mp3')):
                # 构造完整的文件路径
                file_path = os.path.join(root, file)

                # 调用处理函数
                transcription = pipe(file_path)["text"]

                # 保存文件名和转录结果到字典中
                results.append({
                    'filename': os.path.relpath(file_path, start=directory),
                    'transcription': transcription
                })

    json_path = os.path.join(directory, 'transcriptions.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def list_subdirectories(base_path):
    # 获取base_path下的所有项
    entries = os.listdir(base_path)

    # 过滤出目录项
    return [os.path.join(base_path, entry) for entry in entries if os.path.isdir(os.path.join(base_path, entry))]



if __name__ == '__main__':
    # 指定要检查的根目录
    base_path = '/mnt/user/bufan/lzy/prompt4_10.23/input'

    for dir_path in list_subdirectories(base_path):
        process_directory(dir_path)

    pass
