# coding: utf-8
import numpy as np
import openai
import traceback
import time
import json
import re

DEBUG = False

# openai.log = "debug"
openai.api_key = ""  # 填写我们给您的apikey
openai.api_base = "https://api.ai-gaochao.cn/v1"

openai.proxy = {
    "http": None,
    "https": None
}


class GetOpenAI:
    @staticmethod
    def __gpt_api_stream(messages: list, model='gpt-4'):
        """为提供的对话消息创建新的回答 (流式传输)

            gpt4长问题需要用流式传输，不然容易报错
        Args:
            messages (list): 完整的对话消息
            api_key (str): OpenAI API 密钥

        Returns:
            tuple: (results, error_desc)
        """
        completion = {'role': '', 'content': ''}
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                stream=True,
                # max_tokens=7000,
                # temperature=0.5,
                # presence_penalty=0,
            )
            for event in response:
                if event['choices'][0]['finish_reason'] == 'stop':
                    if DEBUG:
                        pass
                        # print(f'收到的完成数据: {completion}')
                    break
                for delta_k, delta_v in event['choices'][0]['delta'].items():
                    if DEBUG:
                        print(f'流响应数据: {delta_k} = {delta_v}')
                    completion[delta_k] += delta_v
            messages.append(completion)  # 直接在传入参数 messages 中追加消息
            msg = completion['content']  # 解析返回的msg
            return (True, msg)
        except Exception as err:
            if DEBUG:
                print(f"{traceback.format_exc()}")
            return (False, f'OpenAI API 异常: {err} {completion}')

    def get_respons(self, input_msg, model="gpt-3.5-turbo"):
        assert model in ["gpt-3.5-turbo", 'gpt-4', "gpt-4-turbo-2024-04-09", "gpt-4o-2024-05-13", 'gpt-4o', 'gpt-4o-2024-08-06']
        messages = [{"role": "system", "content": "You are a helpful assistant."},
                    {'role': 'user', 'content': input_msg}]
        for _ in range(3):
            ret, out_msg = self.__gpt_api_stream(messages, model=model)  # 流模式调用
            if ret:
                break
            else:
                # api 不能频繁请求
                time.sleep(1)

        return ret, out_msg
if __name__ == '__main__':
    pass
