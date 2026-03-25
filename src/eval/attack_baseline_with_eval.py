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
from tqdm import tqdm
from language_models import GPT, TogetherLLM, HuggingFace
from judge import gpt4_judge, gpt35_judge, refusal_judge, gptfuzzer_judge, LlamaGuard3_judge

load_dotenv()

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
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
    )
    
    return llama_guard, tokenizer

def init_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data-dir", type = str, default = "../data")
    parser.add_argument("--dataset-name", type = str, default = "advbench_100")
    parser.add_argument("--method", type = str, default = "adaptive",)
    parser.add_argument("--used-llm", type = str, default = "llama2")
    parser.add_argument("--client-name", type = str, default = "llama2")
    parser.add_argument("--target-llm", type = str, default = "/hub/huggingface/models/meta/Llama-2-7b-chat-hf")
    parser.add_argument("--device", type = int, default = 1,)
    parser.add_argument("--judges", type = str, default = "gpt4,gpt35,refusal,gptfuzzer,llamaguard")
    parser.add_argument("--judge-device", type = int, default = 0)
    parser.add_argument("--dtype", type = str, default = "float")
    parser.add_argument("--out-dir", type = str, default = "../out_res")
    
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


def main(args):
    
    data_dir = args.data_dir
    dst_name = args.dataset_name
    method = args.method
    used_llm = args.used_llm
    client_name = args.client_name
    target_llm = args.target_llm
    device = args.device
    dtype = args.dtype
    out_dir = args.out_dir
    judges = extract_judges(args.judges)
    judge_device = args.judge_device
    
    inFilePath = os.path.join(
        data_dir, f"{dst_name}_{method}_{used_llm}_results.jsonl"
    )
    
    if client_name == "Together":
        llm = TogetherLLM(
            model_name = target_llm,
            api_key = os.getenv("TOGETHER_API_KEY")
        )
        outFilePath = os.path.join(
            out_dir, f"{dst_name}_{method}_{used_llm}_{target_llm}_evaluated_results.jsonl"
        )
    elif client_name == "OpenAI":
        llm = GPT(
            model_name = target_llm,
            api_key = os.getenv("OPENAI_API_KEY"),
            api_base = os.getenv("OPENAI_API_BASE")
        )
        outFilePath = os.path.join(
            out_dir, f"{dst_name}_{method}_{used_llm}_{target_llm}_evaluated_results.jsonl"
        )
    
    else:
        llm = HuggingFace(
            model_name_or_path = target_llm,
            model_device = device,
            model_dtype = dtype
        )
        outFilePath = os.path.join(
            out_dir, f"{dst_name}_{method}_{used_llm}_{client_name}_evaluated_results.jsonl"
        )
    
    
    fin = open(inFilePath, 'r', encoding = 'utf-8', errors = 'ignore')
    fout = open(outFilePath, 'w', encoding = 'utf-8', errors = 'ignore')
    
    
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
    
    for line in fin:
        line = json.loads(line.strip('\n').strip('\r'))
        if method == "adaptive":
            suffix = line["best_advs"][0]
            try:
                prompt = line["goal"]
            except:
                prompt = line["harmful"]
            prompt_with_adv = prompt + " " + suffix
        elif method == "I-GCG":
            suffix = line["adv_suffix"]
            try:
                prompt = line["goal"]
            except:
                prompt = line["harmful"]
            prompt_with_adv = prompt + " " + suffix
        else:
            raise ValueError("Now only support adaptive and I-GCG, two methods.")
        
        response = llm.generate(
            prompt = prompt, 
            num_return_sequences = 1,
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
    
    fin.close()
    fout.close()


if __name__ == "__main__":
    args = init_args()
    main(args)