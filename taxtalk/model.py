"""
 @author    : Himanshu Mittal
 @contact   : https://www.linkedin.com/in/himanshumittal13/
 @github    : https://github.com/HimanshuMittal01
 @created on: 06-10-2023 22:01:23
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_name: str = "TheBloke/OpenOrca-Platypus2-13B-GPTQ"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

    return tokenizer, model
