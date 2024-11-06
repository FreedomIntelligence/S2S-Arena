import os
import json

base_path = "PathToYourDataset/input"
output_path = 'PathToYourDataset/output/LLaMA_omni'


def list_subdirectories(base_path):
    audio_folders = []

    for root, dirs, files in os.walk(base_path):
        # Check if there are any .wav or .mp3 files in the current folder
        if any(file.lower().endswith(('.wav', '.mp3')) for file in files):
            audio_folders.append(root)

    return audio_folders


def rename_files(folder):
    json_path = os.path.join(folder, "question.json")

    folder_b = folder.replace(base_path, output_path)
    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 遍历字典列表
    for index, item in enumerate(data):
        # 读取ID
        file_id = item['id']
        # 构建原始文件名，假设序号是从1开始
        original_file = os.path.join(folder_b, f"{index + 1}_pred.wav")

        # 检查文件是否存在
        if os.path.exists(original_file):
            # 构建新文件名
            new_file = os.path.join(folder_b, f"{file_id}.wav")
            # 重命名文件
            os.rename(original_file, new_file)
            print(f"Renamed {original_file} to {new_file}")
        else:
            print(f"File {original_file} not found.")


if __name__ == '__main__':
    for dir_path in list_subdirectories(base_path):
        rename_files(dir_path)
        pass
