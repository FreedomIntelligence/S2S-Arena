import json
import os
import re
import sys
sys.path.append('PATH_TO_COSYVOICE')

from cosyvoice.cli.cosyvoice import CosyVoice
import torchaudio

cosyvoice = CosyVoice('PATH_TO_COSYVOICE/pretrained_models/CosyVoice-300M-Instruct')


def process_responses(input_dir, output_dir):
    json_path = os.path.join(input_dir, "funaudio_qwen72b_llm_response.json")

    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for item in data:
        full_path = os.path.join(output_dir, item['filename'])
        full_path = full_path.replace(".mp3",".wav")
        os.makedirs(os.path.dirname(full_path),exist_ok=True)
        # 读取 llm_response
        text = item['llm_response']

        pattern = r"生成风格:\s*([^;]+);播报内容:\s*(.+)"
        match = re.search(pattern, text)
        if match:
            style = match.group(1).strip()
            content = match.group(2).strip()
            tts_text = f"{style}<endofprompt>{content}"
            print(f"生成风格: {style}")
            print(f"播报内容: {content}")
        else:
            print("No match found")
            tts_text = text

        output = cosyvoice.inference_sft(tts_text, '中文女')
        torchaudio.save(full_path, output['tts_speech'], 22050)

def list_subdirectories(base_path):
    entries = os.listdir(base_path)

    return [os.path.join(base_path, entry) for entry in entries if os.path.isdir(os.path.join(base_path, entry))]

if __name__ == '__main__':
    base_path = 'PathToYourDataset/input'
    output_path = 'PathToYourDataset/funaudio_guan'

    for dir_path in list_subdirectories(base_path):
        output_dir = dir_path.replace(base_path,output_path)
        process_responses(dir_path,output_dir)

    pass