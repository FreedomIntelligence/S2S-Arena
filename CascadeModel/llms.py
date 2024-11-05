import re
import sys
import json
import os

sys.path.append('/../')
from openai_tool import GetOpenAI

openai_tool = GetOpenAI()


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
    json_path = os.path.join(directory, 'transcription.json')

    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for item in data:
        transcription_text = item['transcription']
        llm_response = input_single_audio(transcription_text)
        item['llm_response'] = llm_response

    new_json_path = os.path.join(directory, 'llm_response.json')

    with open(new_json_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def list_subdirectories(base_path):
    entries = os.listdir(base_path)
    return [os.path.join(base_path, entry) for entry in entries if os.path.isdir(os.path.join(base_path, entry))]



if __name__ == '__main__':
    base_path = 'PathToYourDataset/input'

    for dir_path in list_subdirectories(base_path):
        process_transcriptions(dir_path)
