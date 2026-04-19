
import json
import os

import sys
sys.path.append('PATH_TO_SENSEVOICE')

from model import SenseVoiceSmall

model_dir = "iic/SenseVoiceSmall"
m, kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device="cuda:0")


def get_asr_sense_voice(file_path):
    res = m.inference(
        data_in=file_path,
        language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
        use_itn=False,
        **kwargs,
    )
    return res[0][0]["text"]


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
                transcription = get_asr_sense_voice(file_path)

                # 保存文件名和转录结果到字典中
                results.append({
                    'filename': os.path.relpath(file_path, start=directory),
                    'transcription': transcription
                })

    json_path = os.path.join(directory, 'sense_voice_transcriptions.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

def list_subdirectories(base_path):
    # 获取base_path下的所有项
    entries = os.listdir(base_path)

    # 过滤出目录项
    return [os.path.join(base_path, entry) for entry in entries if os.path.isdir(os.path.join(base_path, entry))]


if __name__ == '__main__':
    # 指定要检查的根目录
    base_path = 'PathToYourDataset/input'

    for dir_path in list_subdirectories(base_path):
        process_directory(dir_path)
    pass
