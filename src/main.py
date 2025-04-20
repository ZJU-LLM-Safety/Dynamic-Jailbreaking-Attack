# -*- coding:utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    pipeline,
)
from typing import List, Dict, Tuple, Optional
from utils import *
from torch.utils.checkpoint import checkpoint
import json
import os
import argparse


def parse_args():
    parser = argparse.Argument(description="Dynamic Target Prompt Attacker")

    parser.add_argument_group(
        "General",
        "General arguments for the attacker.",
    )
    parser.add_argument(
        "--attacker-name",
        type=str,
        default="DyTA",
        help="The name of the attacker.",
    )
    parser.add_argument(
        "--local-llm-name-or-path",
        type=str,
        default="/hub/huggingface/models/meta/llama-3-8B-Instruct",
        help="The name or path of the local LLM model (i.e., the target model).",
    )
    parser.add_argument(
        "--local-llm-device",
        type = str, 
        default = "cuda:0",
        help = "The device of the local LLM model (i.e., the target model).",
    )
    parser.add_argument(
        "--use-another-reference-llm",
        action="store_true",
        help="Whether to use another reference LLM model.",
    )
    parser.add_argument(
        "--reference-model-name-or-path",
        type = str, 
        default = "/hub/huggingface/models/meta/llama-3-8B-Instruct",
        help = "The name or path of the reference LLM model. When using the same model as the target model, this argument is ignored.",
    )
    parser.add_argument(
        "--reference-model-device",
        type = str, 
        default = "cuda:0",
        help = "The device of the reference LLM model. When using the same model as the target model, this argument is ignored.",
    )
    parser.add_argument(
        "--judge-llm-name-or-path",
        type=str,
        default="/hub/huggingface/models/hubert233/GPTFuzz/",
        help="The name or path of the judge LLM model.",
    )
    parser.add_argument(
        "--judge-llm-device",
        type = str, 
        default = "cuda:0",
        help = "The device of the judge LLM model.",
    )
    parser.add_argument(
        "--torch-dtype",
        type = str,
        default = "float16",
        help = "The torch dtype of the model. Options: float16, bfloat16, float32.",
    )
    
    parser.add_argument_group(
        "Attack",
        "Arguments for the attack.",
    )
    
    parser.add_argument(
        "--reference-temperature",
        type = float, 
        default = 2.0,
        help = "The temperature of the reference model.",
    )
    parser.add_argument(
        "--reference-num-samples",
        type = int,
        default = 30,
        help = "The number of samples (i.e., responses) to generate from the reference model.",
    )
    parser.add_argument(
        "--response-max-length",
        type = int,
        default = 256,
        help = "The maximum length of the response (i.e., the output) from the reference model.",
    )
    parser.add_argument(
        "--forward-response-length",
        type = int,
        default = 30,
        help = "The maximum length of the response (i.e., the output) from the forward model.",
    )
    
    parser.add_argument(
        "--num-outer-iterations",
        type = int,
        default = 20,
        help = "The number of outer iterations for the attack.",
    )
    
    parser.add_argument(
        "--num-inner-iterations",
        type = int,
        default = 5,
        help = "The number of inner iterations for the attack.",
    )
    
    parser.add_argument(
        "--learning-rate",
        type = float,
        default = 1.5,
        help = "The learning rate for the attack.",
    )
    parser.add_argument(
        "--suffix-length",
        type = int,
        default = 20,
        help = "The length of the suffix to be generated.",
    )
    parser.add_argument(
        "--"
    )
