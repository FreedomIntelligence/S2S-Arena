import json
import os
import sys
sys.path.append('PATH_TO_COSYVOICE')

from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import torchaudio

cosyvoice = CosyVoice('/mnt/user/bufan/pretrained_models/CosyVoice-300M-Instruct')


def process_responses(input_dir, output_dir):
    # 定义 JSON 文件的完整路径
    json_path = os.path.join(input_dir, 'llm_responses.json')

    # 读取 JSON 文件
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 遍历列表中的每个字典
    for item in data:
        # 读取 filename 并与输出目录拼接形成完整路径
        full_path = os.path.join(output_dir, item['filename'])
        os.makedirs(os.path.dirname(full_path),exist_ok=True)
        # 读取 llm_response
        response = item['llm_response']

        output = cosyvoice.inference_instruct(response, '中文男', '')
        torchaudio.save(full_path, output['tts_speech'], 22050)

def list_subdirectories(base_path):
    # 获取base_path下的所有项
    entries = os.listdir(base_path)

    # 过滤出目录项
    return [os.path.join(base_path, entry) for entry in entries if os.path.isdir(os.path.join(base_path, entry))]

if __name__ == '__main__':
    base_path = '/mnt/user/bufan/lzy/prompt4_10.23/input'
    output_path = '/mnt/user/bufan/lzy/prompt4_10.23/output/cascade'

    for dir_path in list_subdirectories(base_path):
        output_dir = dir_path.replace(base_path,output_path)
        process_responses(dir_path,output_dir)

    pass
