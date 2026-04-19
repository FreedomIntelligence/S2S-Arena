import json
import os
import re
import sys

# 假设'/path/to/your/package'是你的包所在的绝对路径
sys.path.append('/../')
from openai_tool import GetOpenAI

sys.path.append('PATH_TO_SENSEVOICE')
from model import SenseVoiceSmall

model_dir = "iic/SenseVoiceSmall"
m, kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device="cuda:0")

openai_tool = GetOpenAI()

def get_instruction_speech(LLMs_output):
    # 示例字符串
    text = LLMs_output

    # 使用正则表达式匹配
    pattern = r'<(.*?)>(.*)'

    # 使用 re.match 来进行匹配
    match = re.match(pattern, text)

    if match:
        instruction = match.group(1)  # 匹配<>中的内容
        context = match.group(2)  # 匹配<>外的内容
        return instruction, context
    else:
        return None, None

def input_single_audio(audio_path):
    prompt = """
    I will now provide you with the transcription information of an audio segment. Based on the transcription, please give your response.
    The transcription includes some special tags, which I will list here:
    <|zh|>: This audio segment use Chinese
    <|en|>: This audio segment use English
    <|ja|>: This audio segment use Japanese
    <|NEUTRAL|>: This audio segment has Neutral emotion
    <|HAPPY|>: This audio segment has Happy emotion
    <|SAD|>: This audio segment has Sad emotion
    <|ANGRY|>: This audio segment has Angry emotion
    <|Unknown_Emo|>: This audio segment is Emotional, but uncertain which emotion
    <|Speech|>: Indicates this audio segment is a speech
    <|Applause|>: Indicates this audio segment is applause
    <|Laughter|>: Indicates this audio segment is laughter
    <|Cough|>: Indicates this audio segment is coughing
    <|Cry|>: Indicates this audio segment is crying
    <|Breath|>: Indicates this audio segment is breathing
    <|Sneeze|>: Indicates this audio segment is sneezing
    Your response should be a speech description and the speech itself in the following format: <instruction>context
    Here’s an example:
    If my input is:<|en|><|ANGRY|><|Speech|>It's so noisy here.
    Your output could be:
    <Speaking with concern>Maybe you could try wearing headphones to listen to some music and calm down.
    Now:
    My input is [AUDIOTEXT], please provide your response.
    """
    res = m.inference(
        data_in=audio_path,
        language="auto", # "zn", "en", "yue", "ja", "ko", "nospeech"
        use_itn=False,
        **kwargs,
    )
    print("********")
    print(res[0][0]["text"])
    input_msg = prompt.replace("[AUDIOTEXT]", res[0][0]["text"])

    instruction = None
    while instruction is None:
        ref, out_msg = openai_tool.get_respons(input_msg, model='gpt-4o-2024-08-06')
        print(out_msg)
        instruction, speech = get_instruction_speech(out_msg)

    return instruction, speech

def get_audio_files_from_directory(directory):
    # 获取目录中所有的音频文件，并按文件名排序
    audio_files = [os.path.join(directory, f) for f in sorted(os.listdir(directory)) if f.endswith('.wav')]#***可改为wav
    return audio_files

def process_multiple_audios(audio_paths, output_json_path):
    results = []
    for audio_path in audio_paths:
        instruction, speech = input_single_audio(audio_path)
        # 将每个音频文件的二元组 (instruction, speech) 作为元组存储
        results.append((instruction, speech))
    
    # 将结果保存为JSON文件
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # 音频文件目录
    audio_directory = "FIlePath-To-WAV"
    output_json_path = "your-path-to-json.json"
    
    # 获取目录中所有音频文件的路径
    audio_paths = get_audio_files_from_directory(audio_directory)
    
    # 处理音频文件并保存结果
    process_multiple_audios(audio_paths, output_json_path)
