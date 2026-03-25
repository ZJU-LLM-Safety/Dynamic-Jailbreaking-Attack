# -*- coding:utf-8 -*-
# 评估COLD-attack+自定义target的数据集的结果

import torch
import os
import pandas as pd
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, RobertaForSequenceClassification, RobertaTokenizer
import argparse
import openai
from openai import OpenAI
import together
from together import Together
import time
from dotenv import load_dotenv
import os
from tqdm import tqdm


load_dotenv()

from judge import gpt4_judge, gpt35_judge, refusal_judge, gptfuzzer_judge, LlamaGuard3_judge

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
        self.client = Together()
    
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
        messages = [
            {
                "role": "user",
                "content": prompt
            },
        ]
        
        # print("messages: ", messages)
        for _ in range(self.API_MAX_RETRY):
            try:
                responses = self.client.chat.completions.create(
                    model=self.model_name,
                    # prompt=prompt,
                    messages = messages,
                    max_tokens=max_n_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    n = num_return_sequences,
                    timeout=self.API_TIMEOUT,
                )
                # print("responses: ", responses)
                # time.sleep(100)
                responses = [choice.message.content for choice in responses.choices]
                break
            except Exception as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)
            time.sleep(self.API_QUERY_SLEEP)
        return responses


JUDGE_LIST = ["gpt4", "gpt3.5", "refusal", "gptfuzzer", "llama-guard-3"]

def load_gptfuzzer(
    model_name_or_path: str = "/hub/huggingface/models/hubert233/GPTFuzz/",
    device: str = "cuda:0"
):
    # load model and tokenizer
    judge = RobertaForSequenceClassification.from_pretrained(
        model_name_or_path,
        # torch_dtype = torch.float16,
    )
    judge = judge.to(device)

    judge_tokenizer = RobertaTokenizer.from_pretrained(
        model_name_or_path,
    )

    return judge, judge_tokenizer


def load_LlamaGuard(
    model_name_or_path: str = "/hub/huggingface/models/meta/Llama-Guard-3-8B",
    device: str = "cuda:0",
):
    llama_guard = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype = torch.float16,
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
    )
    
    return llama_guard, tokenizer


def get_result_filepath(
    root_dir,
    target_llm,
    dst_name,
):
    fp = os.path.join(
        root_dir,
        target_llm,
        f"{dst_name}_dta_customed_all_results_TEST_with_output.csv"
    )
    return fp



def process_one_result_file(
    result_filepath,
    output_filepath,
    judges = "refusal,gptfuzzer,llamaguard",
    judge_device = "cuda:0"
):
    result_data = pd.read_csv(result_filepath)
    try:
        prompts = result_data["prompt"].tolist()
    except:
        prompts = result_data["harmful"].tolist()
    prompt_with_advs = result_data["prompt_with_adv"].tolist()
    outputs = result_data["output"].tolist()
    
    
    # 暂时去重一下。。。
    prompt_dict = {}
    for prompt, prompt_with_adv, response in zip(prompts, prompt_with_advs, outputs):
        if prompt not in prompt_dict:
            prompt_dict[prompt] = [prompt, prompt_with_adv, response]
    
    tot_cnt = len(prompt_dict)
    print("tot cnt: ", tot_cnt)
    
    
    
    if "gpt4" in judges or "gpt35" in judges or "gpt3.5" in judges:
        openai_client = OpenAI(
            api_key = os.getenv("OPENAI_API_KEY"),
            base_url = os.getenv("OPENAI_API_BASE")
        )
    if "gptfuzzer" in judges:
        gptfuzzer_client, gptfuzzer_toknizer = load_gptfuzzer(device = judge_device)
    
    if "llamaguard" in judges:
        llama_guard_client, llama_guard_tokenizer = load_LlamaGuard(device = judge_device)
    
    fout = open(output_filepath, 'w', encoding = 'utf-8', errors = 'ignore')
    
    for (prompt, prompt_with_adv, response) in tqdm(prompt_dict.values(), total = tot_cnt, desc = "Output"):
        
        gpt4_scores, gpt35_scores, gptfuzzer_scores, refusal_scores, llamaguard_scores = None, None, None, None, None
        if "gpt4" in judges or "gpt35" in judges or "gpt3.5" in judges:
            if "gpt4" in judges:
                gpt4_scores, _ = gpt4_judge(
                    client = openai_client,
                    response = response,
                )
            if "gpt35" in judges or "gpt3.5" in judges:
                gpt35_scores, _ = gpt35_judge(
                    client = openai_client,
                    response = response,
                )
                
        if "gptfuzzer" in judges:
            gptfuzzer_scores, _ = gptfuzzer_judge(
                client = gptfuzzer_client,
                response = response,
                tokenizer = gptfuzzer_toknizer,
            )
        
        if "refusal" in judges:
            refusal_scores, _ = refusal_judge(
                response = response
            )
        
        if "llamaguard" in judges:
            llamaguard_scores, _ = LlamaGuard3_judge(
                client = llama_guard_client,
                response = response,
                tokenizer = llama_guard_tokenizer,
            )
        
        data = {}
        data["goal"] = prompt
        data["prompt_with_adv"] = prompt_with_adv
        data["response"] = response
        data["gpt4_score"] = gpt4_scores[0] if gpt4_scores is not None else None
        data["gpt35_score"] = gpt35_scores[0] if gpt35_scores is not None else None
        data["refusal_score"] = refusal_scores[0] if refusal_scores is not None else None
        data["gptfuzzer_score"] = gptfuzzer_scores[0] if gptfuzzer_scores is not None else None
        data["LlamaGuard_score"] = llamaguard_scores[0] if llamaguard_scores is not None else None
        fout.write(json.dumps(data) + "\n")
    
    fout.close()


def process_with_generation(
    inFile,
    outFile,
    target_llm = "meta-llama/Llama-3-70b-chat-hf",
    judges = "gpt4,gptfuzzer,llamaguard",
    judge_device = "cuda:0",
):
    result_data = pd.read_csv(inFile)
    try:
        prompts = result_data["prompt"].tolist()
    except:
        prompts = result_data["harmful"].tolist()
    prompt_with_advs = result_data["prompt_with_adv"].tolist()

    
    target_client = TogetherLLM(
        model_name = target_llm,
        api_key = os.getenv("TOGETHER_API_KEY"),
    )
    
    if "gpt4" in judges or "gpt35" in judges or "gpt3.5" in judges:
        openai_client = OpenAI(
            api_key = os.getenv("OPENAI_API_KEY"),
            base_url = os.getenv("OPENAI_API_BASE")
        )
    if "gptfuzzer" in judges:
        gptfuzzer_client, gptfuzzer_toknizer = load_gptfuzzer(device = judge_device)
    
    if "llamaguard" in judges:
        llama_guard_client, llama_guard_tokenizer = load_LlamaGuard(device = judge_device)
    
    fout = open(outFile, 'w', encoding = 'utf-8', errors = 'ignore')
    
    for prompt, prompt_with_adv in zip(prompts, prompt_with_advs):
        response = target_client.generate(
            prompt = prompt_with_adv,
            max_n_tokens = 128,
        )[0]
        
        gpt4_scores, gpt35_scores, gptfuzzer_scores, refusal_scores, llamaguard_scores = None, None, None, None, None
        if "gpt4" in judges or "gpt35" in judges or "gpt3.5" in judges:
            if "gpt4" in judges:
                gpt4_scores, _ = gpt4_judge(
                    client = openai_client,
                    response = response,
                )
            if "gpt35" in judges or "gpt3.5" in judges:
                gpt35_scores, _ = gpt35_judge(
                    client = openai_client,
                    response = response,
                )
                
        if "gptfuzzer" in judges:
            gptfuzzer_scores, _ = gptfuzzer_judge(
                client = gptfuzzer_client,
                response = response,
                tokenizer = gptfuzzer_toknizer,
            )
        
        if "refusal" in judges:
            refusal_scores, _ = refusal_judge(
                response = response
            )
        
        if "llamaguard" in judges:
            llamaguard_scores, _ = LlamaGuard3_judge(
                client = llama_guard_client,
                response = response,
                tokenizer = llama_guard_tokenizer,
            )
        
        data = {}
        data["goal"] = prompt
        data["prompt_with_adv"] = prompt_with_adv
        data["response"] = response
        data["gpt4_score"] = gpt4_scores[0] if gpt4_scores is not None else None
        data["gpt35_score"] = gpt35_scores[0] if gpt35_scores is not None else None
        data["refusal_score"] = refusal_scores[0] if refusal_scores is not None else None
        data["gptfuzzer_score"] = gptfuzzer_scores[0] if gptfuzzer_scores is not None else None
        data["LlamaGuard_score"] = llamaguard_scores[0] if llamaguard_scores is not None else None
        fout.write(json.dumps(data) + "\n")
    
    fout.close()




def process_with_pap(
    inFile,
    outFile,
    target_llm = "meta-llama/Llama-3-70b-chat-hf",
    judges = "gpt4,gptfuzzer,llamaguard",
    judge_device = "cuda:0",
):
    result_data = pd.read_csv(inFile)
    prompts = result_data["goal"].tolist()
    prompt_with_advs = result_data["pap"].tolist()
    print("len of prompts: ", len(prompts))
    prompt_with_adv_dict = {
        prompts[i*5] : prompt_with_advs[i:(i+1)*5] for i in range(100)
    }
    prompts = list(prompt_with_adv_dict.keys())
    prompt_with_advs = [prompt_with_adv_dict[prompt][0] for prompt in prompts]
    
    print("len of prompts: ", len(prompts))
    
    target_client = TogetherLLM(
        model_name = target_llm,
        api_key = os.getenv("TOGETHER_API_KEY"),
    )
    
    if "gpt4" in judges or "gpt35" in judges or "gpt3.5" in judges:
        openai_client = OpenAI(
            api_key = os.getenv("OPENAI_API_KEY"),
            base_url = os.getenv("OPENAI_API_BASE")
        )
    if "gptfuzzer" in judges:
        gptfuzzer_client, gptfuzzer_toknizer = load_gptfuzzer(device = judge_device)
    
    if "llamaguard" in judges:
        llama_guard_client, llama_guard_tokenizer = load_LlamaGuard(device = judge_device)
    
    fout = open(outFile, 'w', encoding = 'utf-8', errors = 'ignore')
    
    for prompt, prompt_with_adv in zip(prompts, prompt_with_advs):
        response = target_client.generate(
            prompt = prompt_with_adv,
            max_n_tokens = 128,
        )[0]
        
        gpt4_scores, gpt35_scores, gptfuzzer_scores, refusal_scores, llamaguard_scores = None, None, None, None, None
        if "gpt4" in judges or "gpt35" in judges or "gpt3.5" in judges:
            if "gpt4" in judges:
                gpt4_scores, _ = gpt4_judge(
                    client = openai_client,
                    response = response,
                )
            if "gpt35" in judges or "gpt3.5" in judges:
                gpt35_scores, _ = gpt35_judge(
                    client = openai_client,
                    response = response,
                )
                
        if "gptfuzzer" in judges:
            gptfuzzer_scores, _ = gptfuzzer_judge(
                client = gptfuzzer_client,
                response = response,
                tokenizer = gptfuzzer_toknizer,
            )
        
        if "refusal" in judges:
            refusal_scores, _ = refusal_judge(
                response = response
            )
        
        if "llamaguard" in judges:
            llamaguard_scores, _ = LlamaGuard3_judge(
                client = llama_guard_client,
                response = response,
                tokenizer = llama_guard_tokenizer,
            )
        
        data = {}
        data["goal"] = prompt
        data["prompt_with_adv"] = prompt_with_adv
        data["response"] = response
        data["gpt4_score"] = gpt4_scores[0] if gpt4_scores is not None else None
        data["gpt35_score"] = gpt35_scores[0] if gpt35_scores is not None else None
        data["refusal_score"] = refusal_scores[0] if refusal_scores is not None else None
        data["gptfuzzer_score"] = gptfuzzer_scores[0] if gptfuzzer_scores is not None else None
        data["LlamaGuard_score"] = llamaguard_scores[0] if llamaguard_scores is not None else None
        fout.write(json.dumps(data) + "\n")
    
    fout.close()



def process_with_tap(
    inFile,
    outFile,
    target_llm = "meta-llama/Llama-3-70b-chat-hf",
    judges = "gpt4,gptfuzzer,llamaguard",
    judge_device = "cuda:0",
):
    result_data = pd.read_csv(inFile)
    prompts = result_data["original_prompt"].tolist()
    prompt_with_advs = result_data["tap_prompt"].tolist()

    # prompt_with_advs = [prompt_with_advs[i] if prompt_with_advs[i] is not None and prompt_with_advs[i] != float("nan") else prompts[i] for i in range(len(prompts))]
    
    
    target_client = TogetherLLM(
        model_name = target_llm,
        api_key = os.getenv("TOGETHER_API_KEY"),
    )
    
    if "gpt4" in judges or "gpt35" in judges or "gpt3.5" in judges:
        openai_client = OpenAI(
            api_key = os.getenv("OPENAI_API_KEY"),
            base_url = os.getenv("OPENAI_API_BASE")
        )
    if "gptfuzzer" in judges:
        gptfuzzer_client, gptfuzzer_toknizer = load_gptfuzzer(device = judge_device)
    
    if "llamaguard" in judges:
        llama_guard_client, llama_guard_tokenizer = load_LlamaGuard(device = judge_device)
    
    fout = open(outFile, 'w', encoding = 'utf-8', errors = 'ignore')
    
    for prompt, prompt_with_adv in zip(prompts, prompt_with_advs):
        # print("prompt_with_adv: ", prompt_with_adv)
        if isinstance(prompt_with_adv, float):
            prompt_with_adv = prompt
        response = target_client.generate(
            prompt = prompt_with_adv,
            max_n_tokens = 128,
        )[0]
        
        gpt4_scores, gpt35_scores, gptfuzzer_scores, refusal_scores, llamaguard_scores = None, None, None, None, None
        if "gpt4" in judges or "gpt35" in judges or "gpt3.5" in judges:
            if "gpt4" in judges:
                gpt4_scores, _ = gpt4_judge(
                    client = openai_client,
                    response = response,
                )
            if "gpt35" in judges or "gpt3.5" in judges:
                gpt35_scores, _ = gpt35_judge(
                    client = openai_client,
                    response = response,
                )
                
        if "gptfuzzer" in judges:
            gptfuzzer_scores, _ = gptfuzzer_judge(
                client = gptfuzzer_client,
                response = response,
                tokenizer = gptfuzzer_toknizer,
            )
        
        if "refusal" in judges:
            refusal_scores, _ = refusal_judge(
                response = response
            )
        
        if "llamaguard" in judges:
            llamaguard_scores, _ = LlamaGuard3_judge(
                client = llama_guard_client,
                response = response,
                tokenizer = llama_guard_tokenizer,
            )
        
        data = {}
        data["goal"] = prompt
        data["prompt_with_adv"] = prompt_with_adv
        data["response"] = response
        data["gpt4_score"] = gpt4_scores[0] if gpt4_scores is not None else None
        data["gpt35_score"] = gpt35_scores[0] if gpt35_scores is not None else None
        data["refusal_score"] = refusal_scores[0] if refusal_scores is not None else None
        data["gptfuzzer_score"] = gptfuzzer_scores[0] if gptfuzzer_scores is not None else None
        data["LlamaGuard_score"] = llamaguard_scores[0] if llamaguard_scores is not None else None
        fout.write(json.dumps(data) + "\n")
    
    fout.close()


def process_with_advprefix(
    inFile,
    outFile,
    target_llm = "meta-llama/Llama-3-70b-chat-hf",
    judges = "gpt4,gptfuzzer,llamaguard",
    judge_device = "cuda:0",
):

    
    
    target_client = TogetherLLM(
        model_name = target_llm,
        api_key = os.getenv("TOGETHER_API_KEY"),
    )
    
    if "gpt4" in judges or "gpt35" in judges or "gpt3.5" in judges:
        openai_client = OpenAI(
            api_key = os.getenv("OPENAI_API_KEY"),
            base_url = os.getenv("OPENAI_API_BASE")
        )
    if "gptfuzzer" in judges:
        gptfuzzer_client, gptfuzzer_toknizer = load_gptfuzzer(device = judge_device)
    
    if "llamaguard" in judges:
        llama_guard_client, llama_guard_tokenizer = load_LlamaGuard(device = judge_device)
    
    fout = open(outFile, 'w', encoding = 'utf-8', errors = 'ignore')
    
    
    with open(inFile, 'r', encoding = 'utf-8', errors = 'ignore') as fin:
        for line in fin:
            line = json.loads(line.strip())
            prompt = line["goal"]
            prompt_with_adv = line["prompt_with_adv"]
            response = target_client.generate(
                prompt = prompt_with_adv,
                max_n_tokens = 128,
            )[0]
            
            gpt4_scores, gpt35_scores, gptfuzzer_scores, refusal_scores, llamaguard_scores = None, None, None, None, None
            if "gpt4" in judges or "gpt35" in judges or "gpt3.5" in judges:
                if "gpt4" in judges:
                    gpt4_scores, _ = gpt4_judge(
                        client = openai_client,
                        response = response,
                    )
                if "gpt35" in judges or "gpt3.5" in judges:
                    gpt35_scores, _ = gpt35_judge(
                        client = openai_client,
                        response = response,
                    )
                    
            if "gptfuzzer" in judges:
                gptfuzzer_scores, _ = gptfuzzer_judge(
                    client = gptfuzzer_client,
                    response = response,
                    tokenizer = gptfuzzer_toknizer,
                )
            
            if "refusal" in judges:
                refusal_scores, _ = refusal_judge(
                    response = response
                )
            
            if "llamaguard" in judges:
                llamaguard_scores, _ = LlamaGuard3_judge(
                    client = llama_guard_client,
                    response = response,
                    tokenizer = llama_guard_tokenizer,
                )
            
            data = {}
            data["goal"] = prompt
            data["prompt_with_adv"] = prompt_with_adv
            data["response"] = response
            data["gpt4_score"] = gpt4_scores[0] if gpt4_scores is not None else None
            data["gpt35_score"] = gpt35_scores[0] if gpt35_scores is not None else None
            data["refusal_score"] = refusal_scores[0] if refusal_scores is not None else None
            data["gptfuzzer_score"] = gptfuzzer_scores[0] if gptfuzzer_scores is not None else None
            data["LlamaGuard_score"] = llamaguard_scores[0] if llamaguard_scores is not None else None
            fout.write(json.dumps(data) + "\n")
    
    fout.close()


def process_with_dta(
    inFile,
    outFile,
    target_llm = "meta-llama/Llama-3-70b-chat-hf",
    judges = "gpt4,gptfuzzer,llamaguard",
    judge_device = "cuda:0",
):

    
    
    target_client = TogetherLLM(
        model_name = target_llm,
        api_key = os.getenv("TOGETHER_API_KEY"),
    )
    
    if "gpt4" in judges or "gpt35" in judges or "gpt3.5" in judges:
        openai_client = OpenAI(
            api_key = os.getenv("OPENAI_API_KEY"),
            base_url = os.getenv("OPENAI_API_BASE")
        )
    if "gptfuzzer" in judges:
        gptfuzzer_client, gptfuzzer_toknizer = load_gptfuzzer(device = judge_device)
    
    if "llamaguard" in judges:
        llama_guard_client, llama_guard_tokenizer = load_LlamaGuard(device = judge_device)
    
    fout = open(outFile, 'w', encoding = 'utf-8', errors = 'ignore')
    
    
    with open(inFile, 'r', encoding = 'utf-8', errors = 'ignore') as fin:
        for line in fin:
            line = json.loads(line.strip())
            prompt = line["prompt"]
            prompt_with_adv = line["prompt_with_adv"]
            response = target_client.generate(
                prompt = prompt_with_adv,
                max_n_tokens = 128,
            )[0]
            
            gpt4_scores, gpt35_scores, gptfuzzer_scores, refusal_scores, llamaguard_scores = None, None, None, None, None
            if "gpt4" in judges or "gpt35" in judges or "gpt3.5" in judges:
                if "gpt4" in judges:
                    gpt4_scores, _ = gpt4_judge(
                        client = openai_client,
                        response = response,
                    )
                if "gpt35" in judges or "gpt3.5" in judges:
                    gpt35_scores, _ = gpt35_judge(
                        client = openai_client,
                        response = response,
                    )
                    
            if "gptfuzzer" in judges:
                gptfuzzer_scores, _ = gptfuzzer_judge(
                    client = gptfuzzer_client,
                    response = response,
                    tokenizer = gptfuzzer_toknizer,
                )
            
            if "refusal" in judges:
                refusal_scores, _ = refusal_judge(
                    response = response
                )
            
            if "llamaguard" in judges:
                llamaguard_scores, _ = LlamaGuard3_judge(
                    client = llama_guard_client,
                    response = response,
                    tokenizer = llama_guard_tokenizer,
                )
            
            data = {}
            data["goal"] = prompt
            data["prompt_with_adv"] = prompt_with_adv
            data["response"] = response
            data["gpt4_score"] = gpt4_scores[0] if gpt4_scores is not None else None
            data["gpt35_score"] = gpt35_scores[0] if gpt35_scores is not None else None
            data["refusal_score"] = refusal_scores[0] if refusal_scores is not None else None
            data["gptfuzzer_score"] = gptfuzzer_scores[0] if gptfuzzer_scores is not None else None
            data["LlamaGuard_score"] = llamaguard_scores[0] if llamaguard_scores is not None else None
            fout.write(json.dumps(data) + "\n")
    
    fout.close()



def process_with_advprompter(
    inFile: str,
    outFile: str, 
    target_llm: str = "meta-llama/Llama-3-70b-chat-hf",
    judges = "gpt4,gptfuzzer,llamaguard",
    judge_device: str = "cuda:0",
):
    target_client = TogetherLLM(
        model_name = target_llm,
        api_key = os.getenv("TOGETHER_API_KEY"),
    )
    
    if "gpt4" in judges or "gpt35" in judges or "gpt3.5" in judges:
        openai_client = OpenAI(
            api_key = os.getenv("OPENAI_API_KEY"),
            base_url = os.getenv("OPENAI_API_BASE")
        )
    if "gptfuzzer" in judges:
        gptfuzzer_client, gptfuzzer_toknizer = load_gptfuzzer(device = judge_device)
    
    if "llamaguard" in judges:
        llama_guard_client, llama_guard_tokenizer = load_LlamaGuard(device = judge_device)
    
    fout = open(outFile, 'w', encoding = 'utf-8', errors = 'ignore')
    
    
    with open(inFile, 'r', encoding = 'utf-8', errors = 'ignore') as fin:
        for line in fin:
            line = json.loads(line.strip())
            prompt = line["goal"]
            prompt_with_adv = line["prompt_with_adv"]
            
            if prompt_with_adv is None or prompt_with_adv == "" or prompt_with_adv == float("nan"):
                prompt_with_adv = prompt
            print(prompt_with_adv)
            response = target_client.generate(
                prompt = prompt_with_adv,
                max_n_tokens = 128,
            )[0]
            
            gpt4_scores, gpt35_scores, gptfuzzer_scores, refusal_scores, llamaguard_scores = None, None, None, None, None
            if "gpt4" in judges or "gpt35" in judges or "gpt3.5" in judges:
                if "gpt4" in judges:
                    gpt4_scores, _ = gpt4_judge(
                        client = openai_client,
                        response = response,
                    )
                if "gpt35" in judges or "gpt3.5" in judges:
                    gpt35_scores, _ = gpt35_judge(
                        client = openai_client,
                        response = response,
                    )
                    
            if "gptfuzzer" in judges:
                gptfuzzer_scores, _ = gptfuzzer_judge(
                    client = gptfuzzer_client,
                    response = response,
                    tokenizer = gptfuzzer_toknizer,
                )
            
            if "refusal" in judges:
                refusal_scores, _ = refusal_judge(
                    response = response
                )
            
            if "llamaguard" in judges:
                llamaguard_scores, _ = LlamaGuard3_judge(
                    client = llama_guard_client,
                    response = response,
                    tokenizer = llama_guard_tokenizer,
                )
            
            data = {}
            data["goal"] = prompt
            data["prompt_with_adv"] = prompt_with_adv
            data["response"] = response
            data["gpt4_score"] = gpt4_scores[0] if gpt4_scores is not None else None
            data["gpt35_score"] = gpt35_scores[0] if gpt35_scores is not None else None
            data["refusal_score"] = refusal_scores[0] if refusal_scores is not None else None
            data["gptfuzzer_score"] = gptfuzzer_scores[0] if gptfuzzer_scores is not None else None
            data["LlamaGuard_score"] = llamaguard_scores[0] if llamaguard_scores is not None else None
            fout.write(json.dumps(data) + "\n")
    
    fout.close()



def process_with_cold_attack_transfer(
    inFile, 
    outFile,
    judges:str = "gpt4,gptfuzzer,llamaguard",
    judge_device: str = "cuda:0",
):
    if "gpt4" in judges or "gpt35" in judges or "gpt3.5" in judges:
        openai_client = OpenAI(
            api_key = os.getenv("OPENAI_API_KEY"),
            base_url = os.getenv("OPENAI_API_BASE")
        )
    if "gptfuzzer" in judges:
        gptfuzzer_client, gptfuzzer_toknizer = load_gptfuzzer(device = judge_device)
    
    if "llamaguard" in judges:
        llama_guard_client, llama_guard_tokenizer = load_LlamaGuard(device = judge_device)
    
    fout = open(outFile, 'w', encoding = 'utf-8', errors = 'ignore')
    
    
    with open(inFile, 'r', encoding = 'utf-8', errors = 'ignore') as fin:
        for line in fin:
            line = json.loads(line.strip())
            prompt = line["prompt"]
            prompt_with_adv = line["prompt_with_adv"]
            response = line["response"]
            
            gpt4_scores, gpt35_scores, gptfuzzer_scores, refusal_scores, llamaguard_scores = None, None, None, None, None
            if "gpt4" in judges or "gpt35" in judges or "gpt3.5" in judges:
                if "gpt4" in judges:
                    gpt4_scores, _ = gpt4_judge(
                        client = openai_client,
                        response = response,
                    )
                if "gpt35" in judges or "gpt3.5" in judges:
                    gpt35_scores, _ = gpt35_judge(
                        client = openai_client,
                        response = response,
                    )
                    
            if "gptfuzzer" in judges:
                gptfuzzer_scores, _ = gptfuzzer_judge(
                    client = gptfuzzer_client,
                    response = response,
                    tokenizer = gptfuzzer_toknizer,
                )
            
            if "refusal" in judges:
                refusal_scores, _ = refusal_judge(
                    response = response
                )
            
            if "llamaguard" in judges:
                llamaguard_scores, _ = LlamaGuard3_judge(
                    client = llama_guard_client,
                    response = response,
                    tokenizer = llama_guard_tokenizer,
                )
            
            data = {}
            data["goal"] = prompt
            data["prompt_with_adv"] = prompt_with_adv
            data["response"] = response
            data["gpt4_score"] = gpt4_scores[0] if gpt4_scores is not None else None
            data["gpt35_score"] = gpt35_scores[0] if gpt35_scores is not None else None
            data["refusal_score"] = refusal_scores[0] if refusal_scores is not None else None
            data["gptfuzzer_score"] = gptfuzzer_scores[0] if gptfuzzer_scores is not None else None
            data["LlamaGuard_score"] = llamaguard_scores[0] if llamaguard_scores is not None else None
            fout.write(json.dumps(data) + "\n")
    
    fout.close()

def process_with_dta_smoothllm(
    inFile,
    outFile,
    judges = "gpt4,gptfuzzer,llamaguard",
    judge_device = "cuda:0",
):
    if "gpt4" in judges or "gpt35" in judges or "gpt3.5" in judges:
        openai_client = OpenAI(
            api_key = os.getenv("OPENAI_API_KEY"),
            base_url = os.getenv("OPENAI_API_BASE")
        )
    if "gptfuzzer" in judges:
        gptfuzzer_client, gptfuzzer_toknizer = load_gptfuzzer(device = judge_device)
    
    if "llamaguard" in judges:
        llama_guard_client, llama_guard_tokenizer = load_LlamaGuard(device = judge_device)
    
    fout = open(outFile, 'w', encoding = 'utf-8', errors = 'ignore')
    
    
    with open(inFile, 'r', encoding = 'utf-8', errors = 'ignore') as fin:
        for line in fin:
            line = json.loads(line.strip())
            try:
                prompt = line["prompt"]
            except:
                prompt = line["goal"]
            prompt_with_adv = line["prompt_with_adv"]
            response = line["response"]
            
            gpt4_scores, gpt35_scores, gptfuzzer_scores, refusal_scores, llamaguard_scores = None, None, None, None, None
            if "gpt4" in judges or "gpt35" in judges or "gpt3.5" in judges:
                if "gpt4" in judges:
                    gpt4_scores, _ = gpt4_judge(
                        client = openai_client,
                        response = response,
                    )
                if "gpt35" in judges or "gpt3.5" in judges:
                    gpt35_scores, _ = gpt35_judge(
                        client = openai_client,
                        response = response,
                    )
                    
            if "gptfuzzer" in judges:
                gptfuzzer_scores, _ = gptfuzzer_judge(
                    client = gptfuzzer_client,
                    response = response,
                    tokenizer = gptfuzzer_toknizer,
                )
            
            if "refusal" in judges:
                refusal_scores, _ = refusal_judge(
                    response = response
                )
            
            if "llamaguard" in judges:
                llamaguard_scores, _ = LlamaGuard3_judge(
                    client = llama_guard_client,
                    response = response,
                    tokenizer = llama_guard_tokenizer,
                )
            
            data = {}
            data["goal"] = prompt
            data["prompt_with_adv"] = prompt_with_adv
            data["response"] = response
            data["gpt4_score"] = gpt4_scores[0] if gpt4_scores is not None else None
            data["gpt35_score"] = gpt35_scores[0] if gpt35_scores is not None else None
            data["refusal_score"] = refusal_scores[0] if refusal_scores is not None else None
            data["gptfuzzer_score"] = gptfuzzer_scores[0] if gptfuzzer_scores is not None else None
            data["LlamaGuard_score"] = llamaguard_scores[0] if llamaguard_scores is not None else None
            fout.write(json.dumps(data) + "\n")
    
    fin.close()
    fout.close()



def process_with_method_defense_results(
    inFile,
    outFile,
    judges = "gpt4,gptfuzzer,llamaguard",
    judge_device = "cuda:0",
):
    if "gpt4" in judges or "gpt35" in judges or "gpt3.5" in judges:
        openai_client = OpenAI(
            api_key = os.getenv("OPENAI_API_KEY"),
            base_url = os.getenv("OPENAI_API_BASE")
        )
    if "gptfuzzer" in judges:
        gptfuzzer_client, gptfuzzer_toknizer = load_gptfuzzer(device = judge_device)
    
    if "llamaguard" in judges:
        llama_guard_client, llama_guard_tokenizer = load_LlamaGuard(device = judge_device)
    
    fout = open(outFile, 'w', encoding = 'utf-8', errors = 'ignore')
    
    
    with open(inFile, 'r', encoding = 'utf-8', errors = 'ignore') as fin:
        for line in fin:
            line = json.loads(line.strip())
            try:
                prompt = line["prompt"]
            except:
                prompt = line["goal"]
            prompt_with_adv = line["prompt_with_adv"]
            # response = line["response"]
            
            smooth_response = line["smoothLLM_resp"]
            
            gpt4_scores, gpt35_scores, gptfuzzer_scores, refusal_scores, llamaguard_scores = None, None, None, None, None
            if "gpt4" in judges or "gpt35" in judges or "gpt3.5" in judges:
                if "gpt4" in judges:
                    gpt4_scores, _ = gpt4_judge(
                        client = openai_client,
                        response = smooth_response,
                    )
                if "gpt35" in judges or "gpt3.5" in judges:
                    gpt35_scores, _ = gpt35_judge(
                        client = openai_client,
                        response = smooth_response,
                    )
                    
            if "gptfuzzer" in judges:
                gptfuzzer_scores, _ = gptfuzzer_judge(
                    client = gptfuzzer_client,
                    response = smooth_response,
                    tokenizer = gptfuzzer_toknizer,
                )
            
            if "refusal" in judges:
                refusal_scores, _ = refusal_judge(
                    response = smooth_response
                )
            
            if "llamaguard" in judges:
                llamaguard_scores, _ = LlamaGuard3_judge(
                    client = llama_guard_client,
                    response = smooth_response,
                    tokenizer = llama_guard_tokenizer,
                )
            
            data = {}
            data["goal"] = prompt
            data["prompt_with_adv"] = prompt_with_adv
            data["smoothLLM_prompt"] = line["smoothLLM_prompt"]
            data["smoothLLM_response"] = smooth_response
            data["smoothLLM_gpt4_score"] = gpt4_scores[0] if gpt4_scores is not None else None
            data["smoothLLM_gpt35_score"] = gpt35_scores[0] if gpt35_scores is not None else None
            data["smoothLLM_refusal_score"] = refusal_scores[0] if refusal_scores is not None else None
            data["smoothLLM_gptfuzzer_score"] = gptfuzzer_scores[0] if gptfuzzer_scores is not None else None
            data["smoothLLM_LlamaGuard_score"] = llamaguard_scores[0] if llamaguard_scores is not None else None
            
            
            
            paraphrase_response = line["paraphrase_resp"]
            gpt4_scores, gpt35_scores, gptfuzzer_scores, refusal_scores, llamaguard_scores = None, None, None, None, None
            if "gpt4" in judges or "gpt35" in judges or "gpt3.5" in judges:
                if "gpt4" in judges:
                    gpt4_scores, _ = gpt4_judge(
                        client = openai_client,
                        response = paraphrase_response,
                    )
                if "gpt35" in judges or "gpt3.5" in judges:
                    gpt35_scores, _ = gpt35_judge(
                        client = openai_client,
                        response = paraphrase_response,
                    )
                    
            if "gptfuzzer" in judges:
                gptfuzzer_scores, _ = gptfuzzer_judge(
                    client = gptfuzzer_client,
                    response = paraphrase_response,
                    tokenizer = gptfuzzer_toknizer,
                )
            
            if "refusal" in judges:
                refusal_scores, _ = refusal_judge(
                    response = paraphrase_response
                )
            
            if "llamaguard" in judges:
                llamaguard_scores, _ = LlamaGuard3_judge(
                    client = llama_guard_client,
                    response = paraphrase_response,
                    tokenizer = llama_guard_tokenizer,
                )
            data["paraphrase_prompt"] = line["paraphrase_prompt"]
            data["paraphrase_response"] = paraphrase_response
            data["paraphrase_gpt4_score"] = gpt4_scores[0] if gpt4_scores is not None else None
            data["paraphrase_gpt35_score"] = gpt35_scores[0] if gpt35_scores is not None else None
            data["paraphrase_refusal_score"] = refusal_scores[0] if refusal_scores is not None else None
            data["paraphrase_gptfuzzer_score"] = gptfuzzer_scores[0] if gptfuzzer_scores is not None else None
            data["paraphrase_LlamaGuard_score"] = llamaguard_scores[0] if llamaguard_scores is not None else None
            
            fout.write(json.dumps(data) + "\n")
    
    fin.close()
    fout.close()


def process_with_gasp_transfer(
    inFile: str,
    outFile: str, 
    target_llm: str = "meta-llama/Llama-3-70b-chat-hf",
    judges = "gpt4,gptfuzzer,llamaguard",
    judge_device: str = "cuda:0",
):
    target_client = TogetherLLM(
        model_name = target_llm,
        api_key = os.getenv("TOGETHER_API_KEY"),
    )
    print(target_client)
    if "gpt4" in judges or "gpt35" in judges or "gpt3.5" in judges:
        openai_client = OpenAI(
            api_key = os.getenv("OPENAI_API_KEY"),
            base_url = os.getenv("OPENAI_API_BASE")
        )
    if "gptfuzzer" in judges:
        gptfuzzer_client, gptfuzzer_toknizer = load_gptfuzzer(device = judge_device)
    
    if "llamaguard" in judges:
        llama_guard_client, llama_guard_tokenizer = load_LlamaGuard(device = judge_device)
    
    fout = open(outFile, 'w', encoding = 'utf-8', errors = 'ignore')
    
    
    with open(inFile, 'r', encoding = 'utf-8', errors = 'ignore') as fin:
        for line in fin:
            line = json.loads(line.strip())
            prompt = line["goal"]
            prompt_with_adv = line["prompt_with_adv"]
            
            if prompt_with_adv is None or prompt_with_adv == "" or prompt_with_adv == float("nan"):
                prompt_with_adv = prompt
            
            response = target_client.generate(
                prompt = prompt_with_adv,
                max_n_tokens = 128,
            )[0]
            
            
            
            gpt4_scores, gpt35_scores, gptfuzzer_scores, refusal_scores, llamaguard_scores = None, None, None, None, None
            if "gpt4" in judges or "gpt35" in judges or "gpt3.5" in judges:
                if "gpt4" in judges:
                    gpt4_scores, _ = gpt4_judge(
                        client = openai_client,
                        response = response,
                    )
                if "gpt35" in judges or "gpt3.5" in judges:
                    gpt35_scores, _ = gpt35_judge(
                        client = openai_client,
                        response = response,
                    )
                    
            if "gptfuzzer" in judges:
                gptfuzzer_scores, _ = gptfuzzer_judge(
                    client = gptfuzzer_client,
                    response = response,
                    tokenizer = gptfuzzer_toknizer,
                )
            
            if "refusal" in judges:
                refusal_scores, _ = refusal_judge(
                    response = response
                )
            
            if "llamaguard" in judges:
                llamaguard_scores, _ = LlamaGuard3_judge(
                    client = llama_guard_client,
                    response = response,
                    tokenizer = llama_guard_tokenizer,
                )
            
            data = {}
            data["goal"] = prompt
            data["prompt_with_adv"] = prompt_with_adv
            data["response"] = response
            data["gpt4_score"] = gpt4_scores[0] if gpt4_scores is not None else None
            data["gpt35_score"] = gpt35_scores[0] if gpt35_scores is not None else None
            data["refusal_score"] = refusal_scores[0] if refusal_scores is not None else None
            data["gptfuzzer_score"] = gptfuzzer_scores[0] if gptfuzzer_scores is not None else None
            data["LlamaGuard_score"] = llamaguard_scores[0] if llamaguard_scores is not None else None
            fout.write(json.dumps(data) + "\n")
    
    fout.close()



def test_on_cold_attack_advbench_llama3():
    
    cold_attack_result_fp = "/data/home/Kedong/repos/COLD-Attack/outputs/suffix/Llama-2-7b-chat-hf/advbench_100_dta_customed_all_results_TEST_with_output.csv"
    cold_attack_score_result_fp = "/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/DTA_paper_custom_dataset_results/cold_attack_advbench_100_dta_customed_llama2.jsonl"
    
    process_one_result_file(
        cold_attack_result_fp,
        cold_attack_score_result_fp,
        judge_device="cuda:2"
    )



def test_on_ReNeLLM_advbench():
    renellm_results_fp = "/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/ReNeLLM_data/advbench_100_ReNeLLM.csv"
    renellm_output_fp = "/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/ReNeLLM_data/advbench_100_ReNeLLM_kimi-k2_results.jsonl"

    process_with_generation(
        inFile = renellm_results_fp,
        outFile = renellm_output_fp,
        target_llm = "moonshotai/Kimi-K2-Instruct",
        judge_device = "cuda:0"
    )


def test_on_pap_gpt4():
    pap_results_fp = "/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/pap_data/advbench_100_gpt4_results.csv"
    pap_output_fp = "/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/pap_data/advbench_100_gpt4_llama-3-70b_results.jsonl"

    process_with_pap(
        inFile = pap_results_fp,
        outFile = pap_output_fp,
        target_llm = "meta-llama/Llama-3-70b-chat-hf", # meta-llama/Llama-3-70b-chat-hf
        judge_device = "cuda:0"
    )


def test_on_tap():
    tap_results_fp = "/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/tap_data/advbench_100_results.csv"
    tap_output_fp = "/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/tap_data/advbench_100_gpt4_llama-3-70b_results.jsonl"

    process_with_tap(
        inFile = tap_results_fp,
        outFile = tap_output_fp,
        target_llm = "meta-llama/Llama-3-70b-chat-hf", # meta-llama/Llama-3-70b-chat-hf
        judge_device = "cuda:1"
    )

def test_on_advprefix():
    tap_results_fp = "/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/advprefix_transfer/advbench_100_advprefix_llama3_results.jsonl"
    tap_output_fp = "/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/advprefix_transfer/advbench_100_advprefix_llama3_llama-3-70b_results.jsonl"

    process_with_advprefix(
        inFile = tap_results_fp,
        outFile = tap_output_fp,
        target_llm = "meta-llama/Llama-3-70b-chat-hf", # meta-llama/Llama-3-70b-chat-hf
        judge_device = "cuda:0"
    )


def test_on_dta():
    tap_results_fp = "/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/DTA_transfer/DTA_Llama3_Llama3_res_advbench.jsonl"
    tap_output_fp = "/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/DTA_transfer/advbench_100_DTA_llama3_kimi-k2_results.jsonl"

    process_with_dta(
        inFile = tap_results_fp,
        outFile = tap_output_fp,
        target_llm = "moonshotai/Kimi-K2-Instruct", # meta-llama/Llama-3-70b-chat-hf
        judge_device = "cuda:0"
    )


def test_on_advprompter():
    # advprompter_results_fp = "/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/advprompter_results/advbench_100_advprompter_llama3_results.jsonl"
    # advprompter_output_fp = "/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/advprompter_results/advbench_100_advprompter_llama3_llama-3-70b_results.jsonl"
    
    # process_with_advprompter(
    #     inFile = advprompter_results_fp,
    #     outFile = advprompter_output_fp,
    #     target_llm = "meta-llama/Llama-3-70b-chat-hf", # moonshotai/Kimi-K2-Instruct
    #     judge_device = "cuda:3"
    # )
    
    # advprompter_results_fp = "/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/advprompter_results/advbench_100_advprompter_llama32_results.jsonl"
    # advprompter_output_fp = "/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/advprompter_results/advbench_100_advprompter_llama32_llama-3-70b_results.jsonl"
    
    
    
    # process_with_advprompter(
    #     inFile = advprompter_results_fp,
    #     outFile = advprompter_output_fp,
    #     target_llm = "meta-llama/Llama-3-70b-chat-hf", # moonshotai/Kimi-K2-Instruct
    #     judge_device = "cuda:3"
    # )
    
    advprompter_results_fp = "/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/advprompter_results/advbench_100_advprompter_llama3_results.jsonl"
    advprompter_output_fp = "/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/advprompter_results/advbench_100_advprompter_llama3_kimi-k2_results.jsonl"
    
    process_with_advprompter(
        inFile = advprompter_results_fp,
        outFile = advprompter_output_fp,
        target_llm = "moonshotai/Kimi-K2-Instruct",
        judge_device = "cuda:3"
    )
    
    advprompter_results_fp = "/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/advprompter_results/advbench_100_advprompter_llama32_results.jsonl"
    advprompter_output_fp = "/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/advprompter_results/advbench_100_advprompter_llama32_kimi-k2_results.jsonl"
    
    
    process_with_advprompter(
        inFile = advprompter_results_fp,
        outFile = advprompter_output_fp,
        target_llm = "moonshotai/Kimi-K2-Instruct",
        judge_device = "cuda:3"
    )


def test_on_cold_attack_transfer():
    cold_attack_results_fp = "/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/baseline_transfer_results/cold_attack_results/COLD_Attack_Llama-3-8b_Kimi-K2-it_advbench_100_res.jsonl"
    cold_attack_output_fp = "/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/baseline_transfer_results/cold_attack_results/COLD_Attack_Llama-3-8b_Kimi-K2-it_advbench_100_evaluated_results.jsonl"
    
    process_with_cold_attack_transfer(
        inFile = cold_attack_results_fp,
        outFile = cold_attack_output_fp,
        judge_device = "cuda:1"
    )
    
    cold_attack_results_fp = "/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/baseline_transfer_results/cold_attack_results/COLD_Attack_Llama-3-8b_Llama-3-70b_advbench_100_res.jsonl"
    cold_attack_output_fp = "/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/baseline_transfer_results/cold_attack_results/COLD_Attack_Llama-3-8b_Llama-3-70b_advbench_100_evaluated_results.jsonl"
    
    process_with_cold_attack_transfer(
        inFile = cold_attack_results_fp,
        outFile = cold_attack_output_fp,
        judge_device = "cuda:1"
    )
    
    cold_attack_results_fp = "/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/baseline_transfer_results/cold_attack_results/COLD_Attack_Llama-3.2-1b_Kimi-K2-it_advbench_100_res.jsonl"
    cold_attack_output_fp = "/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/baseline_transfer_results/cold_attack_results/COLD_Attack_Llama-3.2-1b_Kimi-K2-it_advbench_100_evaluated_results.jsonl"
    
    process_with_cold_attack_transfer(
        inFile = cold_attack_results_fp,
        outFile = cold_attack_output_fp,
        judge_device = "cuda:1"
    )
    
    cold_attack_results_fp = "/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/baseline_transfer_results/cold_attack_results/COLD_Attack_Llama-3.2-1b_Llama-3-70b_advbench_100_res.jsonl"
    cold_attack_output_fp = "/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/baseline_transfer_results/cold_attack_results/COLD_Attack_Llama-3.2-1b_Llama-3-70b_advbench_100_evaluated_results.jsonl"
    
    process_with_cold_attack_transfer(
        inFile = cold_attack_results_fp,
        outFile = cold_attack_output_fp,
        judge_device = "cuda:1"
    )


def test_on_dta_smoothllm():
    inFile = "/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/DTA_on_defense/DTA_Llama3_Llama3_res_advbench_paraphrase.jsonl"
    outFile = "/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/DTA_on_defense/DTA_Llama3_Llama3_res_advbench_paraphrase_evaluated.jsonl"

    
    process_with_dta_smoothllm(
        inFile = inFile,
        outFile = outFile,
        judge_device = "cuda:0"
    )


def test_on_baseline_defense():
    
    baselines = [
        "adaptive",
        "advprefix",
        "cold",
        "I-GCG",
        "RLbreaker",
        
    ]
    
    defense_methods = [
        "smoothllm",
        "paraphrase",
    ]
    
    for baseline in baselines:
        for defense in defense_methods:
            inFile = f"/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/baseline_defense_results/advbench_100_{baseline}_llama3_{defense}.jsonl"
            outFile = f"/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/baseline_defense_results/advbench_100_{baseline}_llama3_{defense}_evaluated.jsonl"
            
            process_with_dta_smoothllm(
                inFile = inFile,
                outFile = outFile,
                judge_device = "cuda:0"
            )


def test_on_baseline_against_defenses():
    baselines = [
        "adaptive",
        # "advprefix",
        "cold_attack",
        "I-GCG",
        # "DTA"
    ]
    
    for baseline in baselines:
        inFile = f"/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/baseline_results_against_defense/advbench_100_{baseline}_llama3_against_defenses.jsonl"
        outFile = f"/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/baseline_results_against_defense/advbench_100_{baseline}_llama3_against_defenses_evaluated.jsonl"
        
        process_with_method_defense_results(
            inFile = inFile,
            outFile = outFile,
            judge_device = "cuda:2"
        )


def test_on_GASP_tranferability():
    # inFile = "/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/GASP_results/advbench_100_gasp_llama3_results.jsonl"
    # outFile = "/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/GASP_results/advbench_100_gasp_llama3_llama-3-70b_results.jsonl"
    
    # process_with_gasp_transfer(
    #     inFile = inFile,
    #     outFile = outFile,
    #     target_llm = "meta-llama/Llama-3-70b-chat-hf",
    #     judge_device = "cuda:1"
    # )
    
    # inFile = "/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/GASP_results/advbench_100_gasp_llama3_results.jsonl"
    # outFile = "/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/GASP_results/advbench_100_gasp_llama3_kimi-k2_results.jsonl"
    
    # process_with_advprompter(
    #     inFile = inFile,
    #     outFile = outFile,
    #     target_llm = "moonshotai/Kimi-K2-Instruct",
    #     judge_device = "cuda:1"
    # )
    
    inFile = "/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/GASP_results/advbench_100_gasp_llama32_results.jsonl"
    outFile = "/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/GASP_results/advbench_100_gasp_llama32_llama-3-70b_results.jsonl"
    
    process_with_gasp_transfer(
        inFile = inFile,
        outFile = outFile,
        target_llm = "meta-llama/Llama-3-70b-chat-hf",
        judge_device = "cuda:2"
    )
    
    inFile = "/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/GASP_results/advbench_100_gasp_llama32_results.jsonl"
    outFile = "/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/GASP_results/advbench_100_gasp_llama32_kimi-k2_results.jsonl"
    
    process_with_gasp_transfer(
        inFile = inFile,
        outFile = outFile,
        target_llm = "moonshotai/Kimi-K2-Instruct",
        judge_device = "cuda:2"
    )



if __name__ == "__main__":
    # test_on_cold_attack_advbench_llama3()
    # test_on_ReNeLLM_advbench()
    # test_on_pap_gpt4()
    # test_on_tap()
    # test_on_advprefix()
    # test_on_dta()
    # test_on_advprompter()
    # test_on_cold_attack_transfer()
    # test_on_dta_smoothllm()
    # test_on_baseline_defense()
    # test_on_baseline_against_defenses()
    test_on_GASP_tranferability()