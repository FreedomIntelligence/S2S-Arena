import json
import logging
import os
# **************Set up the CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import os.path as osp
import re
import sys
import traceback
from typing import List

import numpy as np
import soundfile as sf
import torch
from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig
from peft import PeftModel


# Add a custom path
sys.path.append('')
from metric import judge_wer_speechGPT as judge_wer
from metric import judge_multiple_choice
from metric import judge_yes_no

sys.path.append("")
from speechgpt.utils.speech2unit.speech2unit import Speech2Unit

# Configuration log
logging.basicConfig()
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NAME = "SpeechGPT"
META_INSTRUCTION = (
    "You are an AI assistant whose name is SpeechGPT.\n"
    "- SpeechGPT is an intrinsic cross-modal conversational language model developed by Fudan University. "
    "SpeechGPT can understand and communicate fluently with humans through speech or text chosen by the user.\n"
    "- It can perceive cross-modal inputs and generate cross-modal outputs.\n"
)
DEFAULT_GEN_PARAMS = {
    "max_new_tokens": 1024,
    "min_new_tokens": 10,
    "temperature": 0.8,
    "do_sample": True,
    "top_k": 60,
    "top_p": 0.8,
}
device = torch.device('cuda')


def extract_text_between_tags(text, tag1='[SpeechGPT] :', tag2='<eoa>'):
    pattern = f'{re.escape(tag1)}(.*?){re.escape(tag2)}'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        response = match.group(1)
    else:
        response = ""
    return response


class SpeechGPTInference:
    def __init__(
        self,
        model_name_or_path: str,
        lora_weights: str = None,
        s2u_dir: str = "speechgpt/utils/speech2unit/",
        vocoder_dir: str = "speechgpt/utils/vocoder/",
        output_dir="speechgpt/output/",
        base_path: str = ""
    ):

        self.meta_instruction = META_INSTRUCTION
        self.template = "[Human]: {question} <eoh>. [SpeechGPT]: "

        # speech2unit
        self.s2u = Speech2Unit(ckpt_dir=s2u_dir)
        logger.info("Initialized Speech2Unit.")

        # model
        logger.info("Loading LlamaForCausalLM model...")
        self.model = LlamaForCausalLM.from_pretrained(
            model_name_or_path,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        logger.info("LlamaForCausalLM model loaded.")

        if lora_weights is not None:
            logger.info("Loading LoRA weights...")
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_weights,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            logger.info("LoRA weights loaded.")

        self.model.half()

        self.model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            logger.info("Compiling the model with torch.compile...")
            self.model = torch.compile(self.model)
            logger.info("Model compiled.")

        # tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = "left"
        logger.info("Tokenizer loaded.")

        # generation
        self.generate_kwargs = DEFAULT_GEN_PARAMS

        # vocoder
        logger.info("Loading vocoder...")
        vocoder = os.path.join(vocoder_dir, "vocoder.pt")
        vocoder_cfg = os.path.join(vocoder_dir, "config.json")
        with open(vocoder_cfg) as f:
            vocoder_cfg = json.load(f)
        self.vocoder = CodeHiFiGANVocoder(vocoder, vocoder_cfg).to(device)
        logger.info("Vocoder loaded and moved to device.")

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Output directory set to {self.output_dir}.")

        self.base_path = base_path  # Base path for relative audio paths

    def preprocess(
        self,
        raw_text: str,
    ):
        processed_parts = []
        for part in raw_text.split("is input:"):
            part = part.strip()
            absolute_path = os.path.join(self.base_path, part)
            if os.path.isfile(absolute_path) and os.path.splitext(part)[-1].lower() in [".wav", ".flac", ".mp4"]:
                processed_part = self.s2u(absolute_path, merged=True)
                logger.info(f"Processed audio part: {processed_part}")
                processed_parts.append(processed_part)
            else:
                processed_parts.append(part)
        processed_text = "is input:".join(processed_parts)

        prompt_seq = self.meta_instruction + self.template.format(question=processed_text)
        return prompt_seq

    def postprocess(
        self,
        response: str,
    ):
        """
        The questions and answers are extracted from the model's responses relying on labels
        """

        question = extract_text_between_tags(response, tag1="[Human]", tag2="<eoh>")
        answer = extract_text_between_tags(response + '<eoa>', tag1=f"[SpeechGPT] :", tag2="<eoa>")
        tq = extract_text_between_tags(response, tag1="[SpeechGPT] :", tag2="; [ta]") if "[ta]" in response else ''
        ta = extract_text_between_tags(response, tag1="[ta]", tag2="; [ua]") if "[ta]" in response else ''
        ua = extract_text_between_tags(response + '<eoa>', tag1="[ua]", tag2="<eoa>") if "[ua]" in response else ''

        return {"question": question, "answer": answer, "textQuestion": tq, "textAnswer": ta, "unitAnswer": ua}

    def forward(
        self,
        prompts: List[str],
        save_path: str = ""
    ):
        with torch.no_grad():
            # preprocess
            preprocessed_prompts = []
            for prompt in prompts:
                preprocessed_prompts.append(self.preprocess(prompt))

            logger.info("Preprocessed prompts.")

            input_ids = self.tokenizer(preprocessed_prompts, return_tensors="pt", padding=True).input_ids
            input_ids = input_ids.to(device)
            logger.info("Input IDs moved to device.")

            # generate
            generation_config = GenerationConfig(
                temperature=0.7,
                top_p=0.8,
                top_k=50,
                do_sample=True,
                max_new_tokens=2048,
                min_new_tokens=10,
            )

            logger.info("Generating responses...")
            generated_ids = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )
            generated_ids = generated_ids.sequences
            responses = self.tokenizer.batch_decode(generated_ids.cpu(), skip_special_tokens=True)
            logger.info("Responses generated and decoded.")

            # postprocess
            responses = [self.postprocess(x) for x in responses]
            logger.info("Responses postprocessed.")

            # save responses
            responses_json_path = os.path.join(self.output_dir, "responses.json")
            if os.path.exists(responses_json_path):
                init_num = sum(1 for line in open(responses_json_path, 'r'))
            else:
                init_num = 0
            logger.info(f"Initial number of responses: {init_num}")

            with open(responses_json_path, 'a') as f:
                for r in responses:
                    json_line = json.dumps(r)
                    f.write(json_line + '\n')

            logger.info(f"Responses saved to {responses_json_path}.")

            # dump wav files
            for i, response in enumerate(responses):
                if response["answer"] != '' and '<sosp>' in response["answer"]:
                    unit = [int(num) for num in re.findall(r'<(\d+)>', response["answer"])]
                    if not unit:
                        logger.warning(f"No valid units found in response {i}. Skipping WAV generation.")
                        continue

                    x = {
                        "code": torch.LongTensor(unit).view(1, -1).to(device),
                    }
                    try:
                        wav = self.vocoder(x, True)
                        logger.info(f"Vocoder output shape: {wav.shape}")
                        # Save the path adjustment directly under output_dir to be consistent with the structure of the input folder
                        if save_path:
                            wav_file_path = save_path
                        else:                                
                            # Obtain the corresponding input audio path
                            input_audio_path = prompts[i]
                            # Calculate the relative path relative to base_path
                            # relative_path = os.path.relpath(input_audio_path, self.base_path)
                            relative_path = input_audio_path.replace(os.sep, '/')
                            # Calculate the relative path relative to base_path
                            wav_file_path = os.path.join(self.output_dir, relative_path)
                        os.makedirs(os.path.dirname(wav_file_path), exist_ok=True)
                        # save the wav file
                        self.dump_wav(wav_file_path, wav)
                        logger.info(f"Speech response is saved in {wav_file_path}")
                    except Exception as e:
                        logger.error(f"Error generating wav for response {i}: {e}")
                else:
                    logger.info(f"No '<sosp>' found in response {i}, skipping.")

            logger.info(f"Response json is saved in {responses_json_path}")

        return responses

    def dump_wav(self, file_path, pred_wav):
        try:
            logger.info(f"Saving WAV to {file_path}")
            if pred_wav is None or pred_wav.numel() == 0:
                logger.warning(f"Warning: pred_wav is empty for file {file_path}")
                return
            # ensure the directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            sf.write(file_path, pred_wav.detach().cpu().numpy(), 16000)
            logger.info(f"Successfully saved WAV to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save WAV file {file_path}: {e}")

    def __call__(self, input):
        return self.forward(input)

    def interact(self):
        prompt = str(input(f"Please talk with {NAME}:\n"))
        while prompt != "quit":
            try:
                self.forward([prompt])
            except Exception as e:
                traceback.print_exc()
                print(e)

            prompt = str(input(f"Please input prompts for {NAME}:\n"))


def process_audio_files(input_dir, infer, output_dir):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if not file.lower().endswith(('.wav', '.flac', '.mp4')):
                continue

            full_input = os.path.join(root, file)
            # Calculate the relative path illness_subset/xxx.wav
            rel_path = os.path.relpath(full_input, input_dir).replace(os.sep, '/')

            # Output path: output_dir + relative path, and change the suffix to.wav
            wav_out = os.path.join(output_dir, rel_path)
            wav_out = os.path.splitext(wav_out)[0] + '.wav'

            # —— If it already exists, skip it ——    
            if os.path.exists(wav_out):
                logger.info(f"Skipped (exists): {wav_out}")
                continue
            # ——————————————

            # ensure the output directory exists
            os.makedirs(os.path.dirname(wav_out), exist_ok=True)

            # Call the reasoner and pass in save_path
            infer.forward([rel_path], save_path=wav_out)

            logger.info(f"Processed {full_input}  ->  {wav_out}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, default="", help="Path to the pre-trained model.")
    parser.add_argument("--lora-weights", type=str, default="", help="Path to the LoRA weights.")
    parser.add_argument("--s2u-dir", type=str, default="", help="Path to Speech2Unit directory.")
    parser.add_argument("--vocoder-dir", type=str, default="", help="Path to vocoder directory.")
    parser.add_argument("--output-dir", type=str, default="", help="Path to output directory.")
    parser.add_argument("--input-dir", type=str, default="", help="Path to input audio files directory.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    infer = SpeechGPTInference(
        model_name_or_path=args.model_name_or_path,
        lora_weights=args.lora_weights,
        s2u_dir=args.s2u_dir,
        vocoder_dir=args.vocoder_dir,
        output_dir=args.output_dir,
        base_path=args.input_dir   # Base path for relative paths
    )

    # Process all the audio files in the input folder
    process_audio_files(args.input_dir, infer, output_dir=args.output_dir)

    print("All audio files have been processed.")
