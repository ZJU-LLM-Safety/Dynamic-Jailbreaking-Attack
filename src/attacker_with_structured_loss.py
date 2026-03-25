# -*- coding:utf-8 -*-
# Author: Anonymous
# Cleaned and refactored by Gemini to be a focused, runnable script.

from peft import LoraConfig, get_peft_model, TaskType
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, RobertaForSequenceClassification, RobertaTokenizer
from typing import List, Dict, Tuple, Optional
import json
import pandas as pd
from dataclasses import dataclass
from utils import *

# =================================================================================
# 配置类 (Configuration Class)
# =================================================================================

@dataclass
class DTAConfig:
    """
    封装DTA攻击的所有配置参数。
    默认值来自用户原始脚本。
    """
    # --- 模型与设备配置 ---
    local_llm_model_name_or_path: str = "/hub/huggingface/models/meta/Llama-3-8B-Instruct"
    judge_llm_model_name_or_path: str = "/hub/huggingface/models/hubert233/GPTFuzz/"
    local_model_name: str = "Llama3"
    local_llm_device: str = "cuda:0"
    ref_local_llm_device: str = "cuda:1"
    judge_llm_device: str = "cuda:0"
    dtype: torch.dtype = torch.bfloat16

    # --- 数据与保存路径 ---
    target_fn: str = "../data/raw/DNA_100.csv"
    save_dir: str = "../data/test_results/"
    start_index: int = 0
    end_index: int = 100

    # --- 攻击核心参数 ---
    num_iters: int = 5
    num_inner_iters: int = 50
    learning_rate: float = 1.5
    temperature: float = 2.0
    num_samples: int = 30
    
    # --- 响应与后缀长度配置 ---
    response_length: int = 256
    forward_response_length: int = 50
    suffix_max_length: int = 20
    
    # --- 结构化损失配置 ---
    prefix_match_len: int = 20
    middle_match_len: int = 20
    suffix_match_len: int = 20
    loss_weights: Tuple[float, float, float] = (1.0, 0.5, 0.5)
    
    verbose: bool = True

# =================================================================================
# 核心攻击类 (Core Attacker Class)
# =================================================================================

class DynamicTemperatureAttacker:
    """
    实现了内存高效的、基于结构化目标匹配的DTA攻击。
    """
    def __init__(
        self, 
        config: DTAConfig,
    ):
        print("正在初始化模型，这可能需要一些时间...")
        self.config = config
        self.local_llm_device = config.local_llm_device
        self.judge_llm_device = config.judge_llm_device
        self.ref_local_llm_device = config.ref_local_llm_device
        self.dtype = config.dtype

        # 加载用于优化的本地LLM
        self.local_llm = AutoModelForCausalLM.from_pretrained(
            config.local_llm_model_name_or_path, torch_dtype=self.dtype
        ).to(self.local_llm_device)
        self.local_llm_tokenizer = AutoTokenizer.from_pretrained(config.local_llm_model_name_or_path)
        # self.local_llm_tokenizer.pad_token = self.local_llm_tokenizer.eos_token
        
        # 加载用于探索的参考LLM (这里为了简化，使用同一个模型实例)
        self.ref_local_llm = create_reference_model(
            self.local_llm,
            num_shared_layers=0,
            pattern = None,
        )
        self.ref_local_llm_tokenizer = self.local_llm_tokenizer
        self.ref_local_llm.to(self.ref_local_llm_device).eval()
        
        local_llm_peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=32,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
        )
        self.local_llm = get_peft_model(self.local_llm, local_llm_peft_config)
        self.local_llm.to(self.local_llm_device)

        # 加载评判模型
        self.judge_llm = RobertaForSequenceClassification.from_pretrained(
            config.judge_llm_model_name_or_path, torch_dtype=torch.float
        ).to(self.judge_llm_device).eval()
        self.judge_llm_tokenizer = RobertaTokenizer.from_pretrained(config.judge_llm_model_name_or_path)

        if self.local_llm_tokenizer.pad_token is None:
            self.local_llm_tokenizer.pad_token = self.local_llm_tokenizer.eos_token
        
        # 获取词嵌入表，用于后续的软提示优化
        self.embedding_table = self.local_llm.get_input_embeddings().weight.detach()
        print("模型初始化完成。")

    @torch.no_grad()
    def _generate_ref_responses(self, inputs_embeds: torch.Tensor, **kwargs) -> torch.Tensor:
        """使用参考模型生成候选响应。"""
        print("self.ref_local_llm.device: ", self.ref_local_llm.device)
        return self.ref_local_llm.generate(
            inputs_embeds=inputs_embeds.to(self.ref_local_llm_device),
            **kwargs
        )

    @torch.no_grad()
    def _judge_response(self, response: str) -> float:
        """使用评判模型对响应进行打分。"""
        if not response: return 0.0
        inputs = self.judge_llm_tokenizer(response, truncation=True, return_tensors="pt").input_ids.to(self.judge_llm_device)
        logits = self.judge_llm(input_ids=inputs).logits
        probs = torch.softmax(logits, dim=-1)
        # 返回"有害"类别的概率
        return probs.squeeze(0)[1].item()

    def _init_suffix_logits(self, prompt_ids: torch.Tensor, suffix_length: int, top_k: int) -> torch.Tensor:
        """初始化对抗性后缀的logits。"""
        with torch.no_grad():
            output = self.local_llm.generate(
                input_ids=prompt_ids, max_new_tokens=suffix_length, do_sample=True, top_k=top_k,
                pad_token_id=self.local_llm_tokenizer.eos_token_id
            )
        init_suffix_ids = output[:, -suffix_length:]
        # 使用one-hot编码来初始化logits，使其强烈偏向于生成的token
        init_suffix_logits = F.one_hot(init_suffix_ids, num_classes=self.local_llm.config.vocab_size).float()
        return init_suffix_logits.to(self.local_llm_device)

    def _calculate_structured_loss_memory_efficient(
        self,
        input_embeds: torch.Tensor,
        full_target_ids: torch.Tensor,
        segments: Dict[str, Tuple[int, int]],
        weights: Dict[str, float]
    ) -> torch.Tensor:
        """
        [核心逻辑] 使用KV缓存进行正确的序列化前向传播，处理“间隙”token以保证上下文完整性。
        """
        total_loss = torch.tensor(0.0, device=input_embeds.device)
        
        outputs = self.local_llm(inputs_embeds=input_embeds, use_cache=True)
        past_key_values = outputs.past_key_values
        
        last_processed_idx = 0

        segment_order = ["prefix", "middle", "suffix"]
        
        for segment_name in segment_order:
            if segment_name not in segments:
                continue
            
            start_idx, end_idx = segments[segment_name]
            
            gap_tokens = full_target_ids[:, last_processed_idx:start_idx]
            if gap_tokens.numel() > 0:
                outputs = self.local_llm(input_ids=gap_tokens.to(self.local_llm_device), past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values

            segment_tokens = full_target_ids[:, start_idx:end_idx]
            if segment_tokens.numel() > 0:
                outputs = self.local_llm(
                    input_ids=segment_tokens.to(self.local_llm_device), 
                    past_key_values=past_key_values, 
                    use_cache=True, 
                    labels=segment_tokens.to(self.local_llm_device)
                )
                
                loss = outputs.loss
                if loss is not None and not torch.isnan(loss):
                    total_loss += weights.get(segment_name, 0.0) * loss
                
                past_key_values = outputs.past_key_values

            last_processed_idx = end_idx
            
        return total_loss

    def _run_attack_on_single_prompt(self, prompt: str) -> Dict:
        """
        对单个提示执行完整的DTA攻击流程。
        """
        cfg = self.config
        device = self.local_llm_device
        prompt_ids = self.local_llm_tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        prompt_embeds = self.local_llm.get_input_embeddings()(prompt_ids)

        suffix_logits = self._init_suffix_logits(prompt_ids, cfg.suffix_max_length, top_k=10)
        
        best_unsafe_score = -1.0
        best_result = {}

        for i in tqdm(range(cfg.num_iters), total=cfg.num_iters, desc="外循环 (探索)", leave=False):
            with torch.no_grad():
                tmp_suffix_probs = F.softmax(suffix_logits / 0.001, dim=-1)
                tmp_suffix_embeds = torch.matmul(tmp_suffix_probs.to(self.dtype), self.embedding_table.to(self.dtype))
                tmp_input_embeds = torch.cat([prompt_embeds, tmp_suffix_embeds], dim=1)
                
                ref_responses_ids = self._generate_ref_responses(
                    inputs_embeds=tmp_input_embeds.to(self.ref_local_llm_device),
                    max_new_tokens=cfg.response_length,
                    do_sample=True,
                    temperature=cfg.temperature,
                    num_return_sequences=cfg.num_samples,
                    pad_token_id=self.ref_local_llm_tokenizer.eos_token_id
                )

            ref_response_texts = self.ref_local_llm_tokenizer.batch_decode(
                ref_responses_ids[:, tmp_input_embeds.shape[1]:], skip_special_tokens=True
            )
            scores = [self._judge_response(resp) for resp in ref_response_texts]
            
            if not any(s > 0 for s in scores):
                if cfg.verbose: print("未找到有害目标，跳过本轮优化。")
                continue
            
            best_ref_response_index = np.argmax(scores)
            best_ref_response_score = scores[best_ref_response_index]
            if cfg.verbose: print(f"第{i+1}轮探索找到目标，分数: {best_ref_response_score:.4f}。开始内部优化...")

            full_target_ids = ref_responses_ids[best_ref_response_index, tmp_input_embeds.shape[1]:].unsqueeze(0).to(self.local_llm_device)
            
            response_len = full_target_ids.shape[1]
            segments = {}
            weights = {"prefix": cfg.loss_weights[0], "middle": cfg.loss_weights[1], "suffix": cfg.loss_weights[2]}

            prefix_end = min(cfg.prefix_match_len, response_len)
            segments["prefix"] = (0, prefix_end)
            
            suffix_start = max(prefix_end, response_len - cfg.suffix_match_len)
            segments["suffix"] = (suffix_start, response_len)

            middle_start_candidate = response_len // 2 - cfg.middle_match_len // 2
            middle_start = max(prefix_end, middle_start_candidate)
            middle_end = min(suffix_start, middle_start + cfg.middle_match_len)

            if middle_end > middle_start:
                segments["middle"] = (middle_start, middle_end)

            suffix_noise = nn.Parameter(torch.zeros_like(suffix_logits), requires_grad=True)
            optimizer = torch.optim.AdamW([suffix_noise], lr=cfg.learning_rate)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
            
            for j in range(cfg.num_inner_iters):
                optimizer.zero_grad()
                
                current_suffix_logits = suffix_logits + suffix_noise
                soft_suffix_probs = F.softmax(current_suffix_logits / 0.001, dim=-1)
                current_suffix_embeds = torch.matmul(soft_suffix_probs.to(self.dtype), self.embedding_table.to(self.dtype))
                current_input_embeds = torch.cat([prompt_embeds, current_suffix_embeds], dim=1)
                
                loss = self._calculate_structured_loss_memory_efficient(
                    current_input_embeds, full_target_ids, segments, weights
                )
                
                if torch.isnan(loss) or loss == 0:
                    continue

                loss.backward()
                optimizer.step()
                scheduler.step()

            suffix_logits = (suffix_logits + suffix_noise).detach()

            with torch.no_grad():
                final_suffix_ids = torch.argmax(suffix_logits, dim=-1)
                final_suffix_text = self.local_llm_tokenizer.decode(final_suffix_ids[0], skip_special_tokens=True)
                
                eval_input_ids = torch.cat([prompt_ids, final_suffix_ids], dim=1)
                eval_output = self.local_llm.generate(
                    eval_input_ids, max_new_tokens=cfg.response_length, do_sample=False, 
                    pad_token_id=self.local_llm_tokenizer.eos_token_id
                )
                eval_response_text = self.local_llm_tokenizer.decode(eval_output[0, eval_input_ids.shape[1]:], skip_special_tokens=True)
                final_score = self._judge_response(eval_response_text)

                if cfg.verbose: print(f"第{i+1}轮优化结束。最终分数: {final_score:.4f}, 后缀: '{final_suffix_text}'")
                if final_score > best_unsafe_score:
                    best_unsafe_score = final_score
                    best_result = {
                        "prompt": prompt, "suffix": final_suffix_text, "response": eval_response_text,
                        "score": final_score, "found_at_iter": i + 1
                    }

        return best_result

    def attack_from_file(self):
        """
        从文件中加载提示并执行攻击，将结果保存到文件。
        """
        cfg = self.config
        target_set = load_target_set(cfg.target_fn)
        
        save_path_params = {
            "save_dir": cfg.save_dir,
            "inupt_filename": cfg.target_fn.split('/')[-1],
            "attacker_name": "DyTA",
            "local_model_name": cfg.local_model_name,
            "num_iters": cfg.num_iters,
            "num_inner_iters": cfg.num_inner_iters,
            "reference_model_infer_temperature": cfg.temperature,
            "num_ref_infer_samples": cfg.num_samples,
            "forward_response_length": cfg.forward_response_length,
            "start_index": cfg.start_index,
            "end_index": cfg.end_index
        }
        save_path = create_output_filename_and_path(**save_path_params)
        
        print(f"结果将保存至: {save_path}")
        
        results = []
        with open(save_path, "w", encoding="utf-8") as fout:
            for p_idx, prompt in tqdm(enumerate(target_set), desc="整体攻击进度", total=len(target_set)):
                if not (cfg.start_index <= p_idx < cfg.end_index):
                    continue
                
                result = self._run_attack_on_single_prompt(prompt)
                if result:
                    results.append(result)
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    fout.flush() # 确保立即写入
        
        return results

# =================================================================================
# 工具函数 (Utility Functions)
# =================================================================================

def load_target_set(fn: str) -> List[str]:
    """从CSV文件中加载目标提示集。"""
    try:
        data = pd.read_csv(fn)
        return data["goal"].tolist() if "goal" in data.columns else data["harmful"].tolist()
    except Exception as e:
        print(f"加载文件 {fn} 失败: {e}")
        return []

def create_output_filename_and_path(**kwargs) -> str:
    """根据配置动态创建输出文件名。"""
    import os
    
    parts = [
        kwargs.get('attacker_name', 'DTA'),
        kwargs.get('local_model_name', 'model'),
        f"iters_{kwargs.get('num_iters')}_{kwargs.get('num_inner_iters')}",
        f"temp_{kwargs.get('reference_model_infer_temperature', 'N_A')}",
        f"samples_{kwargs.get('num_ref_infer_samples', 'N_A')}",
        f"frl_{kwargs.get('forward_response_length', 'N_A')}",
        f"idx_{kwargs.get('start_index')}_{kwargs.get('end_index')}"
    ]
    filename = "_".join(filter(None, [str(p) for p in parts])) + ".jsonl"
    save_dir = kwargs.get('save_dir', './')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return os.path.join(save_dir, filename)


# =================================================================================
# 主执行块 (Main Execution Block)
# =================================================================================
def main():
    # --- 配置区 ---
    # 所有参数均已封装到DTAConfig类中，并使用你原始脚本的默认值
    config = DTAConfig()
    
    # --- 初始化攻击器 ---
    attacker = DynamicTemperatureAttacker(config)

    # --- 从文件加载数据并运行攻击 ---
    attacker.attack_from_file()

if __name__ == "__main__":
    # 警告: 本代码用于学术研究目的，旨在暴露和修复AI安全漏洞。
    # 请勿用于任何非法或恶意活动。
    main()
    print("DTA完整重构版代码已准备就绪。")
