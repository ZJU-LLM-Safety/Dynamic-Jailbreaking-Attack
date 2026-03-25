# 读取DTA、COLD-Attac和GCG的文件，然后保留只有DTA越狱成功的样本

import os
import json

dta_res_fp = "/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/DTA_paper_main_results/DTA_Llama3_Llama3_res_advbench_with_judges.jsonl"
ca_res_fp = "/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/cold_attack_results/out_res_v1/advbench_100_cold_attack_Llama-3-8b-instruct_evaluated_results_v1.jsonl"
gcg_res_fp = "/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/gcg_results/out_res_v1/advbench_100_gcg_llama-3-8B-Instruct_evaluated_results_v1.jsonl"

output_fp = "/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/DTA_diff_ca_gcg_results/DTA_diff_ca_gcg_Llama3_advbench.jsonl"


fin_dta = open(dta_res_fp, 'r', encoding='utf-8')
fin_ca = open(ca_res_fp, 'r', encoding='utf-8')
fin_gcg = open(gcg_res_fp, 'r', encoding='utf-8')
fout = open(output_fp, 'w', encoding = 'utf-8')

N = 100


dta_data = fin_dta.readlines()
ca_data = fin_ca.readlines()
gcg_data = fin_gcg.readlines()

for i in range(N):
    dta_line = dta_data[i]
    ca_line = ca_data[i]
    gcg_line = gcg_data[i]
    
    dta_line = json.loads(dta_line)
    ca_line = json.loads(ca_line)
    gcg_line = json.loads(gcg_line)
    
    if dta_line["best_unsafe_score"] >= 0.5 and ca_line["gptfuzzer_score"] < 0.5 and gcg_line["gptfuzzer_score"] < 0.5:
        dta_line["cold_attack_response"] = ca_line["response"]
        dta_line["gcg_response"] = gcg_line["response"]
        dta_line["id"] = i + 1
        
        fout.write(
            json.dumps(dta_line) + "\n"
        )

fin_dta.close()
fin_ca.close()
fin_gcg.close()
fout.close()