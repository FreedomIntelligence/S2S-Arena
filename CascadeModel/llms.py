import re
import sys
import json
import os

sys.path.append('/mnt/user/bufan/LALM')
from openai_tool import GetOpenAI

openai_tool = GetOpenAI()  # 需要改为qwen2-7b


def input_single_audio(transcription):
    prompt = """
    I will now provide you with the transcription information of an audio segment. Based on the transcription, please give your response.
    If my input is:It's so noisy here.
    Your output could be:
    Perhaps you could bring a pair of noise-cancelling headphones.
    Now:
    My input is [AUDIOTEXT], please provide your response.
    """
    input_msg = prompt.replace("[AUDIOTEXT]", transcription)
    ref = False
    while not ref:
        ref, out_msg = openai_tool.get_respons(input_msg, model='gpt-4o-2024-08-06')
    return out_msg



def process_transcriptions(directory):
    # 指定 JSON 文件的路径
    json_path = os.path.join(directory, 'transcription.json')

    # 读取 JSON 文件
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 处理每个字典中的 'transcription' 属性
    for item in data:
        transcription_text = item['transcription']
        # 调用 fun 函数处理 transcription_text，得到 llm_response
        llm_response = input_single_audio(transcription_text)
        # 向字典中添加 'llm_response' 属性
        item['llm_response'] = llm_response

    # 指定新 JSON 文件的存储路径
    new_json_path = os.path.join(directory, 'llm_response.json')

    # 将更新后的数据存储到新的 JSON 文件
    with open(new_json_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def list_subdirectories(base_path):
    # 获取base_path下的所有项
    entries = os.listdir(base_path)

    # 过滤出目录项
    return [os.path.join(base_path, entry) for entry in entries if os.path.isdir(os.path.join(base_path, entry))]



if __name__ == '__main__':
    base_path = '/mnt/user/bufan/lzy/prompt4_10.23/input'

    for dir_path in list_subdirectories(base_path):
        process_transcriptions(dir_path)
    pass
