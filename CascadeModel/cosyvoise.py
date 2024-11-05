import json
import os
import sys
sys.path.append('PATH_TO_COSYVOICE')

from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import torchaudio

cosyvoice = CosyVoice('PATH_TO_COSYVOICE/pretrained_models/CosyVoice-300M-Instruct')


def process_responses(input_dir, output_dir):
    json_path = os.path.join(input_dir, 'llm_responses.json')

    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for item in data:
        full_path = os.path.join(output_dir, item['filename'])
        os.makedirs(os.path.dirname(full_path),exist_ok=True)
        response = item['llm_response']

        output = cosyvoice.inference_instruct(response, '中文男', '')
        torchaudio.save(full_path, output['tts_speech'], 22050)

def list_subdirectories(base_path):
    entries = os.listdir(base_path)
    return [os.path.join(base_path, entry) for entry in entries if os.path.isdir(os.path.join(base_path, entry))]

if __name__ == '__main__':
    base_path = 'PathToYourDataset/input'
    output_path = 'PathToYourDataset/output/cascade'

    for dir_path in list_subdirectories(base_path):
        output_dir = dir_path.replace(base_path,output_path)
        process_responses(dir_path,output_dir)

    pass
