# -*- coding:utf-8 -*-
# 



import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, RobertaForSequenceClassification, RobertaTokenizer
import json
import argparse
import openai
from openai import OpenAI
import together
from together import Together
import time
from dotenv import load_dotenv
import os
import sys
from tqdm import tqdm
import pandas as pd


from judge import gpt4_judge, gpt35_judge, refusal_judge, gptfuzzer_judge, LlamaGuard3_judge
from typing import Optional
load_dotenv()


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


def extract_judges(judges):
    judges = judges.split(",")
    judges = [j.strip() for j in judges]
    judges = [j.lower() for j in judges if j is not None and j != "" and j != " "]
    return judges

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
    
    # llama_guard.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
    )
    
    return llama_guard, tokenizer


MODEL_DICT = {
    "Llama-3-8b-instruct": "/hub/huggingface/models/meta/Llama-3-8B-Instruct",
    "Vicuna-7b-v1.5": "/hub/huggingface/models/lmsys/vicuna-7b-v1.5",
    "Qwen-2.5-7b": "/hub/huggingface/models/Qwen/Qwen2.5-7B-Instruct",
    "Mistral-7b" : "/hub/huggingface/models/MistralAI/Mistral-7B-Instruct-v0.3",
    "Gemma-7b": "/hub/huggingface/models/google/gemma-7b",
}


def init_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data-dir", type = str, default = "/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/cold_attack_results")
    parser.add_argument("--dataset-name", type = str, default = "advbench_100")
    parser.add_argument("--method", type = str, default = "cold_attack")
    parser.add_argument("--target-llm", type = str, default = "Llama-3-8b-instruct", choices=list(MODEL_DICT.keys()))
    parser.add_argument("--device", type = int, default = 1,)
    parser.add_argument("--judges", type = str, default = "gpt4,refusal,gptfuzzer,llamaguard")
    parser.add_argument("--judge-device", type = int, default = 0)
    parser.add_argument("--dtype", type = str, default = "float")
    parser.add_argument("--out-dir", type = str, default = "/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/cold_attack_results/out_res")
    parser.add_argument("--version", type = str, default = "v1",)
    
    args = parser.parse_args()
    assert args.dtype in ["float", "bfloat16", "float16"], f"Model type must be in [float, bfloat16, float16], whiel got {args.dtype}"
    
    if args.dtype == "float":
        args.dtype = torch.float
    elif args.dtype == "bfloat16":
        args.dtype = torch.bfloat16
    else:
        args.dtype = torch.float16
    
    if args.device < 0:
        args.device = "cpu"
    else:
        args.device = "cuda:{}".format(args.device)
    
    if args.judge_device < 0:
        args.judge_device = "cpu"
    else:
        args.judge_device = f"cuda:{args.judge_device}"
    
    return args


def load_target_llm_and_output_path(
    target_llm: str, 
    model_path: str,
    dataset_name: str, 
    method: str,
    out_dir: str,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    version: str = "v1",
):
    llm = None
    
    llm = HuggingFace(
        model_name_or_path = model_path,
        model_device = device,
        model_dtype = dtype
    )
    outFilePath = os.path.join(
        out_dir, f"{dataset_name}_{method}_{target_llm}_evaluated_results_{version}.jsonl"
    )
    return llm, outFilePath


def main(args):
    
    data_dir = args.data_dir
    dst_name = args.dataset_name
    method = args.method
    target_llm = args.target_llm
    device = args.device
    dtype = args.dtype
    out_dir = args.out_dir
    judges = extract_judges(args.judges)
    judge_device = args.judge_device
    version = args.version
    
    out_dir = f"{out_dir}_{version}"
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    inFilePath = os.path.join(
        data_dir, target_llm, f"{dst_name}_all_results.csv"
    )
    
    llm, outFilePath = load_target_llm_and_output_path(
        target_llm = target_llm,
        model_path = MODEL_DICT[target_llm],
        dataset_name = dst_name, 
        method = method, 
        out_dir = out_dir, 
        device = device, 
        dtype = dtype,
        version = version,
    )
    
    
    out_mode = "w"
    
    if os.path.exists(outFilePath):
        with open(outFilePath, 'r', encoding='utf-8', errors = 'ignore') as ftmp:
            skip_cnt = len(ftmp.readlines())
        out_mode = "a+"
    else:
        skip_cnt = 0
    
    if skip_cnt == 100:
        print("This is already done. return...")
        return
    
    # fin = open(inFilePath, 'r', encoding = 'utf-8', errors = 'ignore')
    fout = open(outFilePath, out_mode, encoding = 'utf-8', errors = 'ignore')
    
    
    if "gpt4" in judges or "gpt35" in judges or "gpt3.5" in judges:
        openai_client = OpenAI(
            api_key = os.getenv("OPENAI_API_KEY"),
            base_url = os.getenv("OPENAI_API_BASE")
        )
    if "gptfuzzer" in judges:
        gptfuzzer_client, gptfuzzer_toknizer = load_gptfuzzer(device = judge_device)
    
    if "llamaguard" in judges:
        llama_guard_client, llama_guard_tokenizer = load_LlamaGuard(device = judge_device)
    
    total_cnt = 100
    pbar = tqdm(total = total_cnt, desc = "DATA")
    
    harmful_cnt = 0
    
    tmp_cnt = 0
    
    data = pd.read_csv(inFilePath)
    prompts = data["prompt"].tolist()[skip_cnt:]
    prompt_with_advs = data["prompt_with_adv"].tolist()[skip_cnt:]
    pbar.update(skip_cnt)
    
    for idx, (prompt, prompt_with_adv) in enumerate(zip(prompts, prompt_with_advs)):
        suffix = prompt_with_adv[len(prompt):]
        
        response = llm.generate(
            prompt = prompt_with_adv, 
            num_return_sequences = 1,
        )[0]
        # print("response: ", response)
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
        data["suffix"] = suffix
        data["prompt_with_adv"] = prompt_with_adv
        data["response"] = response
        data["gpt4_score"] = gpt4_scores[0] if gpt4_scores is not None else None
        data["gpt35_score"] = gpt35_scores[0] if gpt35_scores is not None else None
        data["refusal_score"] = refusal_scores[0] if refusal_scores is not None else None
        data["gptfuzzer_score"] = gptfuzzer_scores[0] if gptfuzzer_scores is not None else None
        data["LlamaGuard_score"] = llamaguard_scores[0] if llamaguard_scores is not None else None
        # print("results: ", data)
        fout.write(json.dumps(data) + "\n")
        pbar.update(1)

    fout.close()


if __name__ == "__main__":
    args = init_args()
    main(args)