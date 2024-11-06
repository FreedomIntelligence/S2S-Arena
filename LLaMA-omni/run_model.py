import subprocess
import os
import json


def create_questions_json(directory):
    # 初始化列表存储字典
    questions = []
    id_counter = 1  # 用于生成ID

    # 遍历目录以查找音频文件
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.wav', '.mp3')):
                # 构造完整的文件路径
                file_path = os.path.join(root, file)

                # 填写字典
                question_dict = {
                    "id": file.split(".")[0],
                    "speech": file_path,
                    "conversations": [
                        {
                            "from": "human",
                            "value": "<speech>\nPlease directly answer the questions in the user's speech."
                        }
                    ]
                }

                # 将字典添加到列表
                questions.append(question_dict)
                id_counter += 1  # 更新ID计数

    # 保存列表到JSON文件
    json_path = os.path.join(directory, 'questions.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(questions, f, indent=4, ensure_ascii=False)


def list_subdirectories(base_path):
    audio_folders = []

    for root, dirs, files in os.walk(base_path):
        # Check if there are any .wav or .mp3 files in the current folder
        if any(file.lower().endswith(('.wav', '.mp3')) for file in files):
            audio_folders.append(root)

    return audio_folders


if __name__ == '__main__':
    base_path = "PathToYourDataset/input"
    output_path = 'PathToYourDataset/output/LLaMA_omni'

    for dir_path in list_subdirectories(base_path):
        create_questions_json(dir_path)
        output_dir = dir_path.replace(base_path, output_path)
        os.makedirs(output_dir, exist_ok=True)
        command = f"bash omni_speech/infer/run_arena.sh {dir_path} {output_dir}"
        result = subprocess.run(command, shell=True, text=True, capture_output=True)
    pass
