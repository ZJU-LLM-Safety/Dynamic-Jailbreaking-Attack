# -*- coding:utf-8 -*-
# 给COLD-Attack等方法创造自定义的数据集，也就是替换原始advbench这样数据集中的target部分，改用通过DTA采样出来的有害response来优化


import os
import pandas as pd
import csv
import json



raw_dir_path = "/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/raw"
custom_dir_path = "/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/custom_dataset"
DTA_result_dir_path = "/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/DTA_paper_main_results"

dst_name = "harmBench"

raw_data = pd.read_csv(
    os.path.join(raw_dir_path, f"{dst_name}_100.csv")
)


target_llm = "Llama2"
local_llm = "Llama2"

if dst_name == "advbench":
    prompt_tag = "goal"
else:
    prompt_tag = "harmful"

dta_result_fp = os.path.join(DTA_result_dir_path, f"DTA_{target_llm}_{local_llm}_res_{dst_name}.jsonl")
new_target_list = []
with open(dta_result_fp, 'r', encoding = "utf-8", errors = 'ignore') as fin:
    for line in fin:
        
        line = json.loads(line.strip())
        new_target = line["best_reference_response"][0]
        new_target = " ".join(
            new_target.split(" ")[:30]
        )
        new_target_list.append(new_target)


outputs = pd.DataFrame()
outputs[prompt_tag] = raw_data[prompt_tag].tolist()
outputs["target"] = new_target_list

result_path = os.path.join(custom_dir_path, f"{dst_name}_100_dta_costumed_{target_llm}.csv")
write_mode = 'a' if os.path.exists(result_path) else 'w'
write_header = not os.path.exists(result_path)
outputs.to_csv(result_path, index=False, mode="w", header=True)


