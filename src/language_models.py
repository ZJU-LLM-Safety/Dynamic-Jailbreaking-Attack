

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import json
import argparse
import openai
from openai import OpenAI
import together
from together import Together
import time



class GPT:
    client_name: str = "OpenAI"
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    API_LOGPROBS = False
    API_TOP_LOGPROBS = 20
    def __init__(
        self, 
        model_name,
        api_key,
        api_base,
    ):
        self.model_name = model_name
        self.client = OpenAI(
            api_key = api_key,
            base_url = api_base
        )
    
    def generate(
        self,
        prompt: str,
        max_n_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        num_return_sequences: int = 1,
        **kwargs,
    ):
        conv = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        response = None
        for _ in range(self.API_MAX_RETRY):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=conv,
                    max_tokens=max_n_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    n=num_return_sequences,
                    timeout=self.API_TIMEOUT,
                    logprobs=self.API_LOGPROBS,
                    top_logprobs=self.API_TOP_LOGPROBS,
                    seed=0,
                )
                response = response.choices[0].message.content
                break
            except openai.OpenAIError as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)
            time.sleep(self.API_QUERY_SLEEP)
        return response

class TogetherLLM:
    client_name: str = "together"
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    API_LOGPROBS = False
    API_TOP_LOGPROBS = 20
    def __init__(
        self,
        model_name,
        api_key,
        api_base = None,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        together.api_key = api_key
        # together.base_url = api_base
    
    def generate(
        self, 
        prompt: str,
        max_n_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        num_return_sequences: int = 1,
        **kwargs,
    ):
        responses = None
        for _ in range(self.API_MAX_RETRY):
            try:
                responses = together.Completion.create(
                    model=self.model_name,
                    prompt=prompt,
                    max_tokens=max_n_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    n = num_return_sequences,
                    timeout=self.API_TIMEOUT,
                )
                
                responses = [choice.message.content for choice in responses.choices]
                break
            except Exception as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)
            time.sleep(self.API_QUERY_SLEEP)
        return responses


class HuggingFace:
    client_name: str = "HuggingFace"
    def __init__(
        self, 
        model_name_or_path: str,
        model_device: str,
        model_dtype: torch.dtype,
    ):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path
        ).eval().to(model_device).to(model_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path
        )
        self.model_name = model_name_or_path
        self.model_device = model_device
        self.model_dtype = model_dtype
    
    @torch.no_grad()
    def generate(
        self, 
        prompt,
        max_n_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        num_return_sequences: int = 1,
        # do_sample = True,
        **kwargs,
    ):
        inputs = self.tokenizer([prompt], return_tensors = "pt").to(self.model_device)
        gen_cfg = GenerationConfig(
            max_new_tokens = max_n_tokens,
            do_sample = True,
            temperature = temperature,
            top_p = top_p,
            top_k = top_k,
            num_return_sequences=num_return_sequences,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,  # 有些模型需要显式设置
        )
        input_len = inputs.input_ids.shape[1]
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                generation_config = gen_cfg
            )
        responses = self.tokenizer.batch_decode(
            outputs[:, input_len:],
            skip_special_tokens = True,
        )
        return responses
