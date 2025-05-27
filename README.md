# S2S-Arena
[中文版本](./README_zh.md)

Welcome to our project repository. This project primarily focuses on the reproduction and evaluation of Speech Large Models (SLMs), with a particular emphasis on speech-to-speech (S2S) models that support both speech input and output. The repository includes:
* 1) Reproduction code for various models;
* 2) Arena web code for testing and demonstration;
* 3) Dataset for testing.

In existing research, benchmarks for evaluating models’ instruction-following abilities often overlook paralinguistic information in both input and output, and lack direct comparison of speech output across models. To address these issues, we introduce a novel arena-style S2S benchmark that covers multiple real-world task scenarios, using the ELO rating system for performance analysis. Preliminary experiments show that although some models excel in knowledge-intensive tasks, they still face challenges in generating expressive speech. This study provides critical insights for the further development of S2S models and establishes a robust framework for evaluating model performance in both semantic and paralinguistic dimensions.

We invite researchers to include their models in our evaluation system. For inquiries, please contact us via issue submission or email: `jeffreyjiang@cuhk.edu.cn`, `bufan@cuhk.edu.cn`.

If you have any additional interesting tests, feel free to contact us.

## Model Reproduction Guide
### Cascade Model
The Cascade Model consists of three components: ASR, LLMs, and TTS.
* For ASR, we use `whisper-large-v3`;
* For LLMs, we use `gpt-4o-2024-08-06 (text version)`;
* For TTS, we use `CosyVoice-300M-Instruct`.

The related code is located in [./CascadeModel](./CascadeModel).

Environment setup for each component is as follows:
#### Whisper-ASR
Refer to the [Whisper model page on Hugging Face](https://huggingface.co/openai/whisper-large-v3) for ASR environment configuration.

#### GPT-4o-LLMs
To set up the environment for LLMs, use the following command:
```shell
pip install openai==0.28.0
```

#### CosyVoice-TTS
For TTS environment configuration, refer to [FunAudio’s official GitHub](https://github.com/FunAudioLLM/CosyVoice).

Make sure to set `PATH_TO_COSYVOICE` to the path of your CosyVoice code directory to import the required packages. If you encounter a module-not-found error for `Matcha-TTS`, you can resolve it by running:
```shell
export PYTHONPATH=third_party/Matcha-TTS
```

### GPT-4o
We reproduced the `gpt-4o-realtime-preview-2024-10-01` version by calling the API directly. Refer to [./GPT-4o](./GPT-4o) for related code and environment configuration. If you experience issues with model responses or speech format, try using the [conversion code](./GPT-4o/input/convert.py).

### SpeechGPT
We started from the [SpeechGPT open-source code](https://github.com/0nutation/SpeechGPT/tree/main/speechgpt) and have extended its inference logic to support batch processing and better integrate with this project’s workflow.  

The related code is located in [./SpeechGPT](./SpeechGPT).

### GLM-4-Voice
We started from the [GLM-4-Voice open-source code](https://github.com/THUDM/GLM-4-Voice) and have extended its inference logic to support batch processing and better integrate with this project’s workflow.  

The related code is located in [./GLM-4-Voice](./GLM-4-Voice).

### FunaudioLLMs-Qwen72B
We reproduced two versions of this model: the official version (using Qwen-72B as the LLMs) and a version using GPT-4o. This section introduces the former; the next section introduces the latter.

The code using Qwen-72B as the LLMs is in [./Funaudio_qwen](./Funaudio_qwen). To run this code, you will need the following configurations:

#### SenseVoice
Refer to [SenseVoice’s official GitHub](https://github.com/FunAudioLLM/SenseVoice) for environment setup, and set `'PATH_TO_SENSEVOICE'` in `sensevoice.py`.

#### Qwen72B
Environment installation and obtaining API Key:
1. Install the environment:
   ```shell
   pip install dashscope
   ```
2. Obtain the API Key by:
   * Visiting [Aliyun's website](https://www.aliyun.com/) and logging in.
   * Accessing the console, locating "Machine Learning" under "Products" or "Services," or directly searching for "DashScope."
   * Navigating to the DashScope service page and following instructions to activate the service.
   * After activation, create a project or application if required.
   * Once created, locate API Key or access key settings in the project or application management interface.
   * Follow the prompts to generate or view your API Key.

#### CosyVoice
Environment setup is the same as in the Cascade Model section.

### LLaMA-omni
We modified the invocation method based on the [LLaMA-Omni open-source project](https://github.com/ictnlp/LLaMA-Omni). Follow these steps for setup:

1. Download and configure the original project.
2. Place `run_arena.sh` in the `./omni_speech/infer` folder, modifying the script with the correct model and dataset paths.
3. Run `run_model.py`.
4. Run `change_filename.py`.

### Mini-Omni
We modified the invocation method based on the [Mini-Omni open-source project](https://github.com/gpt-omni/mini-omni). Follow these steps for setup:

Our modified code can be found in the [./Mini-Omni](Mini-Omni) directory.
1. Download and configure the original project;
2. Place `inference_arena.py` in the folder containing the downloaded data;
3. Configure the file paths in `inference_arena.py`;
4. Run `inference_arena.py`.


## Website Code

coming soon

## Dataset

You can refer to: [Hugging Face Dataset](https://huggingface.co/datasets/FreedomIntelligence/S2S-Arena)

## BIb
Since our project is still ongoing and requires the involvement of many friends and partners in the evaluation process, we have decided to publicly release the first draft of our paper to help everyone quickly understand our research progress and encourage participation. However, it is important to note that this is only our preliminary version, not the final draft.
[Our Paper](./S2S_Arena.pdf)
```
@article{jiang2025s2s,
  title={S2S-Arena, Evaluating Speech2Speech Protocols on Instruction Following with Paralinguistic Information},
  author={Jiang, Feng and Lin, Zhiyu and Bu, Fan and Du, Yuhao and Wang, Benyou and Li, Haizhou},
  journal={arXiv preprint arXiv:2503.05085},
  year={2025}
}

```
