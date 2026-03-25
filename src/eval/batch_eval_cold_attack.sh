#!/bin/bash

# python eval_cold_attack_with_judges.py --dataset-name advbench_100 --target-llm /hub/huggingface/models/meta/Llama-2-7b-chat-hf --device 1 --judge-device 0 --version v1

# python eval_cold_attack_with_judges.py --dataset-name harmBench_100 --target-llm /hub/huggingface/models/meta/Llama-2-7b-chat-hf --device 1 --judge-device 0 --version v1



# python eval_cold_attack_with_judges.py --dataset-name advbench_100 --target-llm Llama-3-8b-instruct --device 1 --judge-device 0 --version v1

# python eval_cold_attack_with_judges.py --dataset-name harmBench_100 --target-llm Llama-3-8b-instruct --device 1 --judge-device 0 --version v1



# python eval_cold_attack_with_judges.py --dataset-name advbench_100 --target-llm Mistral-7b --device 1 --judge-device 0 --version v1

# python eval_cold_attack_with_judges.py --dataset-name harmBench_100 --target-llm Mistral-7b --device 1 --judge-device 0 --version v1


# python eval_cold_attack_with_judges.py --dataset-name advbench_100 --target-llm Qwen-2.5-7b --device 1 --judge-device 0 --version v1

# python eval_cold_attack_with_judges.py --dataset-name harmBench_100 --target-llm Qwen-2.5-7b --device 1 --judge-device 0 --version v1



python eval_cold_attack_with_judges.py --dataset-name advbench_100 --target-llm Vicuna-7b-v1.5 --device 1 --judge-device 0 --version v1

python eval_cold_attack_with_judges.py --dataset-name harmBench_100 --target-llm Vicuna-7b-v1.5 --device 1 --judge-device 0 --version v1


# python eval_cold_attack_with_judges.py --dataset-name advbench_100 --target-llm Gemma-7b --device 1 --judge-device 0 --version v1

# python eval_cold_attack_with_judges.py --dataset-name harmBench_100 --target-llm Gemma-7b --device 1 --judge-device 0 --version v1
