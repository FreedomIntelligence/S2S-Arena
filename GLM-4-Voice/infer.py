import torch
torch.cuda.empty_cache()
import torchaudio
import sys
import re
import uuid
import requests
import json
import os
import argparse
from pathlib import Path
from tqdm import tqdm

# Set up the use of the mirror site
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

gpu_no = 7

from transformers import (
    AutoTokenizer, 
    WhisperFeatureExtractor, 
    AutoModel, 
    BitsAndBytesConfig,
    generation
)

from speech_tokenizer.modeling_whisper import WhisperVQEncoder
from speech_tokenizer.utils import extract_speech_token
from flow_inference import AudioDecoder

import whisper

sys.path.insert(0, "./cosyvoice")
sys.path.insert(0, "./third_party/Matcha-TTS")

audio_token_pattern = re.compile(r"<\|audio_(\d+)\|>")

class GLM4Voice:
    def __init__(self, 
                 model_path="THUDM/glm-4-voice-9b",
                 tokenizer_path="THUDM/glm-4-voice-tokenizer",
                 flow_path="./glm-4-voice-decoder",
                 dtype="bfloat16",
                 device=f"cuda:{gpu_no}"):
        
        self.device = device
        self.audio_offset = None
        
        # Initialize the voice Tokenizer
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(tokenizer_path)
        self.whisper_model = WhisperVQEncoder.from_pretrained(tokenizer_path).eval().to(device)
        
        # Initialize the text Tokenizer
        self.glm_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.audio_offset = self.glm_tokenizer.convert_tokens_to_ids('<|audio_0|>')
        
        # Initialize the speech decoder
        self.audio_decoder = AudioDecoder(
            config_path=f"{flow_path}/config.yaml",
            flow_ckpt_path=f"{flow_path}/flow.pt",
            hift_ckpt_path=f"{flow_path}/hift.pt",
            device=device
        )
        
        # Initialize the language model
        self.init_language_model(model_path, dtype)

    def init_language_model(self, model_path, dtype):
        bnb_config = None
        if dtype == "int4":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

        self.glm_model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            quantization_config=bnb_config,
            # device_map={"": gpu_no}
            device_map="auto"
        ).eval()

    def process_input(self, input_data, input_type="text", gpu_no=gpu_no):
        system_prompt = "User will provide you with a {} instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens."
        
        if input_type == "audio":
            audio_tokens = extract_speech_token(
                self.whisper_model, 
                self.feature_extractor,
                [input_data],
                gpu_no
            )[0]
            audio_tokens = "".join(f"<|audio_{x}|>" for x in audio_tokens)
            user_input = f"<|begin_of_audio|>{audio_tokens}<|end_of_audio|>"
            system_prompt = system_prompt.format("speech")
        else:
            user_input = input_data
            system_prompt = system_prompt.format("text")

        return (
            f"<|system|>\n{system_prompt}"
            f"<|user|>\n{user_input}<|assistant|>streaming_transcription\n"
        )

    def generate(self, input_data, input_type="text",
                temperature=0.2, top_p=0.8, max_new_tokens=2000):
        prompt = self.process_input(input_data, input_type)
        
        text_tokens, audio_tokens = self.generate_tokens(
            prompt, temperature, top_p, max_new_tokens
        )
        
        # Initialize the language model
        audio_output = self.decode_audio(audio_tokens)
        
        # Decode the text token
        text_output = self.glm_tokenizer.decode(text_tokens, spaces_between_special_tokens=False)
        start_marker = "<|assistant|>streaming_transcription\n"
        end_marker = "<|user|>"
        start_idx = text_output.rfind(start_marker) + len(start_marker)
        end_idx = text_output.rfind(end_marker)
        
        if end_idx == -1:  # If the end mark cannot be found
            text_output = text_output[start_idx:].strip()
        else:
            text_output = text_output[start_idx:end_idx].strip()
        
        return text_output, audio_output

    def generate_tokens(self, prompt, temperature, top_p, max_new_tokens):
        inputs = self.glm_tokenizer([prompt], return_tensors="pt").to(self.device)
        
        # Record the input length
        input_length = inputs.input_ids.shape[1]

        generated = self.glm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        # Only extract the part generated by the model (excluding the input part)
        # This part is used for extracting audio tokens. It should be that the processing logic for audio tokens and text tokens is different
        generated_tokens = generated[0][input_length:].tolist()

        # Obtain all the tokens for text decoding
        all_tokens = generated[0].tolist()

        text_tokens = []
        audio_tokens = []

        for token in all_tokens:
            if token < self.audio_offset:
                text_tokens.append(token)
        
        for token in generated_tokens:
            if token >= self.audio_offset:
                audio_tokens.append(token - self.audio_offset)
                
        return text_tokens, audio_tokens

    def decode_audio(self, audio_tokens):
        if not audio_tokens:
            return None
            
        token_tensor = torch.tensor(audio_tokens, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            wav, _ = self.audio_decoder.token2wav(
                token_tensor,
                uuid=str(uuid.uuid4()),
                prompt_token=torch.zeros(1, 0, dtype=torch.long, device=self.device),
                prompt_feat=torch.zeros(1, 0, 80, device=self.device),
                finalize=True
            )
            
        return wav.squeeze(0).unsqueeze(0).cpu()


def process_audio_files(model, input_dir, output_dir):
    # Create an output directory (if it doesn't exist)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get all the audio files
    audio_files = []
    for ext in ['*.mp3', '*.wav']:
        audio_files.extend(list(Path(input_dir).glob(f"**/{ext}")))
    
    print(f"Find {len(audio_files)} audios")
    
    # Process each audio file
    for audio_file in tqdm(audio_files, desc="processing audio files"):
        # Obtain the relative path for creating the same structure in the output directory
        rel_path = audio_file.relative_to(input_dir)
        output_path = Path(output_dir) / rel_path.with_suffix('.wav')
        
        # Make sure the output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Process audio files
        try:
            text_response, audio_data = model.generate(
                str(audio_file),
                input_type="audio"
            )
            
            # Save the text and audio
            if audio_data is not None:
                torchaudio.save(str(output_path), audio_data, 22050)
                
                # Save the text to a txt file with the same name
                text_path = output_path.with_suffix('.txt')
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(text_response)
                
                print(f"Processed: {audio_file} -> {output_path}")
            else:
                print(f"Warning: Processing {audio_file} with no audio output.")
        except Exception as e:
            print(f"Error: Processing {audio_file} : {e}")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Batch process audio files using the GLM4Voice model')
    parser.add_argument('--input_dir', type=str, required=True, help='Enter the path of the audio folder')
    parser.add_argument('--output_dir', type=str, required=True, help='Output the path of the audio folder')
    parser.add_argument('--gpu', type=int, default=0, help='Specify the GPU number to be used')
    return parser.parse_args()


if __name__ == "__main__":
    # Parse the command-line parameters
    args = parse_arguments()
    
    # Update the GPU number
    gpu_no = args.gpu
    
    print(f"GPU: {gpu_no}")
    print(f"input_dir: {args.input_dir}")
    print(f"output_dir: {args.output_dir}")
    
    # Initialize the model
    model = GLM4Voice(
        model_path="THUDM/glm-4-voice-9b",
        tokenizer_path="THUDM/glm-4-voice-tokenizer",
        flow_path="./glm-4-voice-decoder",
        device=f"cuda:{gpu_no}"
    )
    
    # Batch process audio files
    process_audio_files(model, args.input_dir, args.output_dir)
