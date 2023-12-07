"""
Apply the LoRA weights on top of a base model.

Usage:
python merge_lora.py --base-model ~/model_weights/llama-7b --target-model-path ~/model_weights/baize-7b --lora-path project-baize/baize-lora-7B

"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from datetime import datetime

# model = AutoModelForCausalLM.from_pretrained(model_params['model_id'], torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
# model = PeftModel.from_pretrained(model, model_params['adapter'], device_map='auto', torch_dtype=torch.bfloat16)
# model = model.merge_and_unload()  

def apply_lora(base_model, target_model_path, lora_path):
    print(f"Loading the base model from {base_model}")
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, trust_remote_code=True)
    base_tokenizer = AutoTokenizer.from_pretrained(lora_path)

    print(f"Loading the LoRA adapter from {lora_path}")

    lora_model = PeftModel.from_pretrained(
        base,
        lora_path,
        torch_dtype=torch.bfloat16
    )

    print("Applying the LoRA")
    model = lora_model.merge_and_unload()

    print(f"Saving the target model to {target_model_path}")
    model.save_pretrained(target_model_path)
    base_tokenizer.save_pretrained(target_model_path)

    log_path = os.path.join(target_model_path, 'merge_log.txt')
    with open(log_path, 'w') as f:
        f.write(f'--base-model : {base_model}\n')
        f.write(f'--lora-path : {lora_path}\n')
        f.write(f'--target-model-path : {target_model_path}\n')
        f.write(f'(creation date : {datetime.now()})')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--lora-path", type=str, required=True)

    args = parser.parse_args()

    apply_lora(args.base_model, args.target_model_path, args.lora_path)