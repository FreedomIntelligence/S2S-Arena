from rtclient import InputAudioTranscription, RTClient, RTInputItem, RTOutputItem, RTResponse
import asyncio
import os
import time
import json
from tqdm import tqdm
from azure.core.credentials import AzureKeyCredential
from client_sample_custom import (
    send_audio,
    receive_messages,
    run,
    get_env_var,
    with_openai
)

# Function to process files in a batch
async def process_batch(input_folder: str, output_folder: str, instructions_file: str):
    # Load configuration from JSON file
    with open('config.json', 'r', encoding='utf-8') as file:
        config = json.load(file)
    # Get model name and key from config
    model = config['model_name']
    key = config['key']

    # Create output directory if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all files in the input directory
    files = os.listdir(input_folder)

    with tqdm(total=len(files), desc="Processing Files", unit="file") as pbar:
        for filename in files:
            output_file=filename.split(".")[0]
            file_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, output_file)

            # Skip processing if output already exists (checkpoint feature)
            if os.path.exists(output_path):
                pbar.update(1)
                pbar.set_postfix_str(f"Skipping already processed file: {output_file}")
                continue

            pbar.set_postfix_str(f"Processing file: {filename}")
            start_time = time.time()

            # Create output directory for the specific file
            file_output_dir = os.path.join(output_folder, output_file)
            if not os.path.exists(file_output_dir):
                os.makedirs(file_output_dir)

            # Start processing the file with the model
            try:
                async with RTClient(key_credential=AzureKeyCredential(key), model=model) as client:
                    await run(client, file_path, instructions_file, file_output_dir)
            except Exception as e:
                pbar.set_postfix_str(f"Failed to process {filename}: {e}")
                continue

            end_time = time.time()
            elapsed_time = end_time - start_time
            pbar.set_postfix_str(f"Finished processing {filename} in {elapsed_time:.2f} seconds")
            pbar.update(1)

# Main function to run batch processing
async def main():
    input_folder = "batch_input/natural_audios/"
    output_folder = "batch_output/natural_audios/"
    instructions_file = "system_prompt.txt"

    await process_batch(input_folder, output_folder, instructions_file)

if __name__ == "__main__":
    asyncio.run(main())