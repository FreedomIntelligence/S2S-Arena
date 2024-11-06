# S2S-Bench
[中文版本](./README_zh.md)

Welcome to our project repository. This project focuses on the replication and evaluation of Speech Large Models (SLMs), particularly Speech-to-Speech (S2S) models that support both voice input and output. The repository includes:
* 1) Replication code for each model;
* 2) Arena web page code for testing and demonstration;
* 3) Test datasets.

In current research, the assessment of model command adherence capabilities often overlooks paralinguistic information in inputs and outputs and lacks direct comparisons of voice outputs between models. To address these issues, we introduce a novel arena-style S2S benchmark that covers multiple real-world tasks and uses the ELO rating system for performance analysis. Preliminary experiments indicate that while some models excel in knowledge-intensive tasks, they still face significant challenges in generating expressive speech. This research provides key insights for the further development of S2S models and establishes a robust framework for assessing model performance in semantic and paralinguistic dimensions.

We invite researchers to include your models in our testing framework. If you have any questions, please feel free to raise an issue or contact us by email at: `jeffreyjiang@cuhk.edu.cn`, `bufan@cuhk.edu.cn`.

## Model Replication Guide
### Cascade Model
The Cascade Model consists of three parts: ASR, LLMs, and TTS
* For ASR, we use `whisper-large-v3`;
* For LLMs, we use `gpt-4o-2024-08-06 (text version)`;
* For TTS, we use `CosyVoice-300M-Instruct`.

You can find the relevant code in the [./CascadeModel](./CascadeModel) directory.

Environment setup for each part is as follows:
#### Whisper-ASR
For ASR environment setup, please refer to [Whisper model card on Huggingface](https://huggingface.co/openai/whisper-large-v3);
#### GPT-4o-LLMs
For LLMs environment setup, use the following command:
```shell
pip install openai==0.28.0
```
#### CosyVoice-TTS
For TTS environment setup, please refer to [FunAudio official GitHub](https://github.com/FunAudioLLM/CosyVoice).

Ensure that `PATH_TO_COSYVOICE` is set to your CosyVoice code folder path for successful package import. If you encounter an error indicating the `Matcha-TTS` module cannot be found, you can solve it with the following command:
```shell
export PYTHONPATH=third_party/Matcha-TTS
```

### GPT-4o
We replicated `gpt-4o-realtime-preview-2024-10-01` directly via API. You can find our code and environment setup method at [./GPT-4o](./GPT-4o). If the model does not respond or there are issues with the audio format, you may try using the [conversion code](./GPT-4o/input/convert.py).

### SpeechGPT
We are using the complete [open-source SpeechGPT code](https://github.com/0nutation/SpeechGPT/tree/main/speechgpt) without any modifications.

### FunaudioLLMs-Qwen72B
We have replicated two versions of this model; the relevant code is located at [./Funaudio_qwen](./Funaudio_qwen). To run this code, you will need:

#### SenseVoice
For environment setup, refer to [SenseVoice official GitHub](https://github.com/FunAudioLLM/SenseVoice), and set `'PATH_TO_SENSEVOICE'` in `sensevoice.py`.
#### Qwen72B
Install the environment and obtain your API-Key with:
```shell
pip install dashscope
```
#### CosyVoice
Environment setup is the same as for the Cascade Model.

### LLaMA-omni
We modified the call method based on the [open-source LLaMA-Omni project](https://github.com/ictnlp/LLaMA-Omni). Follow these steps for setup:

1. Download and set up the original project;
2. Place `run_arena.sh` into the `./omni_speech/infer` folder of the original project, adjusting the script to point to your model cache address and dataset path;
3. Run `run_model.py`;
4. Run `change_filename.py`.
