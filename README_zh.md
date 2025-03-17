# S2S-Arena
[For English Version](./README.md)

欢迎访问我们的项目仓库。本项目主要聚焦于语音大模型（SLMs）的复现与评估，尤其关注支持语音输入输出的语音-语音（S2S）模型。仓库内容包括：
* 1) 各模型的复现代码；
* 2) Arena网页代码，用于测试和展示；
* 3) 测试用的数据集。

在现有研究中，模型的指令遵循能力评估常忽略输入与输出的语音信息，并缺少直接比较不同模型的语音输出。为解决这些问题，我们引入了一个新颖的竞技场风格的S2S基准测试，涵盖多个真实场景任务，并采用ELO评级系统进行性能分析。初步实验显示，尽管某些模型在知识密集型任务上表现出色，但在生成具有表现力的语音方面仍面临挑战。此研究为S2S模型的进一步发展提供了关键见解，并建立了一个评估模型在语义和副语言层面性能的稳健框架。

我们诚邀研究人员将您的模型纳入我们的测试体系。若有疑问，欢迎通过提交issue或邮箱联系我们：`jeffreyjiang@cuhk.edu.cn`，`bufan@cuhk.edu.cn`。

如果您有更多有趣的测试，也欢迎您联系我们。

## 模型复现指南
### Cascade Model
Cascade Model由三部分构成：ASR、LLMs、TTS
* ASR部分选用 `whisper-large-v3`；
* LLMs部分选用 `gpt-4o-2024-08-06(文本版)`；
* TTS部分选用 `CosyVoice-300M-Instruct`。

相关代码位于[./CascadeModel](./CascadeModel)目录下。

每个部分的环境配置如下：
#### Whisper-ASR
ASR环境配置请参照[Huggingface的whisper模型页面](https://huggingface.co/openai/whisper-large-v3)；
#### GPT-4o-LLMs
LLMs环境配置命令：
```shell
pip install openai==0.28.0
```
#### CosyVoice-TTS
TTS环境配置请参照[FunAudio官方GitHub](https://github.com/FunAudioLLM/CosyVoice)。

请确保将`PATH_TO_COSYVOICE`设置为您的CosyVoice代码文件夹路径，以便导入所需的包。若遇到`Matcha-TTS`模块找不到的报错，可通过以下命令解决：
```shell
export PYTHONPATH=third_party/Matcha-TTS
```

### GPT-4o
我们直接通过API复现了`gpt-4o-realtime-preview-2024-10-01`版本。相关代码和环境配置方法请参见[./GPT-4o](./GPT-4o)。若出现模型不回答或语音格式问题，可尝试使用[转化代码](./GPT-4o/input/convert.py)。

### SpeechGPT
我们使用了[SpeechGPT开源代码](https://github.com/0nutation/SpeechGPT/tree/main/speechgpt)，未进行修改。

### FunaudioLLMs-Qwen72B
我们复现了该模型的两个版本，分别为官方版本(使用Qwen-72B作为其中的LLMs)和使用GPT-4o的版本，本节介绍前者，下一节介绍后者。

使用Qwen-72B作为其中的LLMs的代码位于[./Funaudio_qwen](./Funaudio_qwen)。为运行此代码，您需要：

#### SenseVoice
配置环境请参照[SenseVoice官方GitHub](https://github.com/FunAudioLLM/SenseVoice)，并在`sensevoice.py`中设置`'PATH_TO_SENSEVOICE'`。
#### Qwen72B
安装环境及获取API-Key:
安装环境使用:
```shell
pip install dashscope
```
获取API-Key参考:
* 1.访问[阿里云官网](https://www.aliyun.com/),并登录。
* 2.进入控制台，找到“产品”或“服务”列表中的“机器学习”或者直接搜索“DashScope”。
* 3.点击进入 DashScope 服务页面，根据页面指引完成相关服务的开通。
* 4.在服务开通后，您可能需要创建一个项目或应用，具体取决于服务的要求。
* 5.创建完成后，在项目或应用的管理界面中找到 API Key 或者访问密钥的相关设置。
* 6.按照提示操作生成或查看您的 API Key。
* 
#### CosyVoice
环境配置同Cascade模型。

### LLaMA-omni
我们在[开源LLaMA-Omni项目](https://github.com/ictnlp/LLaMA-Omni)基础上调整了调用方法。配置运行步骤如下：

1. 下载并配置原项目；
2. 将`run_arena.sh`放入`./omni_speech/infer`文件夹下，修改脚本中的模型路径和数据集路径；
3. 运行`run_model.py`；
4. 运行`change_filename.py`。

### Mini-Omni
我们在[开源Mini-Omni项目](https://github.com/gpt-omni/mini-omni)基础上调整了调用方法。配置运行步骤如下：

我们修改后的代码可以在[./Mini-Omni](Mini-Omni)当中找到。
1. 下载并配置原项目；
2. 将`inference_arena.py`放入下载的数据的文件夹；
3. 配置 `inference_arena.py`当中的文件路径；
4. 运行 `inference_arena.py`。


## 网站代码

coming soon

## 数据集

您可以参考：[hugging face 数据集](https://huggingface.co/datasets/FreedomIntelligence/S2S-Arena)

## BIb

```
@article{jiang2025s2s,
  title={S2S-Arena, Evaluating Speech2Speech Protocols on Instruction Following with Paralinguistic Information},
  author={Jiang, Feng and Lin, Zhiyu and Bu, Fan and Du, Yuhao and Wang, Benyou and Li, Haizhou},
  journal={arXiv preprint arXiv:2503.05085},
  year={2025}
}
```
