import json
import os

import torchaudio

sys.path.append('PATH_TO_COSYVOICE')
from cosyvoice_funaudio_gpt4o.cli.cosyvoice import CosyVoice


def generate_audio_from_json(json_path, output_dir, cosyvoice_model_path):
    # 加载CosyVoice模型
    cosyvoice = CosyVoice(cosyvoice_model_path)

    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 遍历每个二元组 (instruction, speech)
    for idx, (instruction, speech) in enumerate(data):
        # 将 instruction 包裹在尖括号中，并将其作为指令传递给模型，但不包含在最终的生成文本中
        instruct_text = speech  # 只包含实际要说的话
        #instruct_info = f"<{instruction}>"  # 作为额外信息传递给模型
        #print(instruct_info)
        #print(instruct_text)
        #text = f"'{instruct_info}{instruct_text}'"
        # 调用CosyVoice生成语音
        output = cosyvoice.inference_instruct(
            instruct_text, 
            '英文男', 
            instruction,
        )
        
        # 保存生成的音频
        output_path = os.path.join(output_dir, f'audio_{idx}.wav')
        torchaudio.save(output_path, output['tts_speech'], 22050)
        print(f"Audio saved at: {output_path}")

if __name__ == "__main__":
    json_path = "your-path-to-json.json"
    output_dir = "YOUR_DIR"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    cosyvoice_model_path = 'pretrained_models/CosyVoice-300M-Instruct'
    
    generate_audio_from_json(json_path, output_dir, cosyvoice_model_path)
