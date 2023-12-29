"""
Load and save model from a checkpoint

Usage:
python save_model.py --checkpoint-path project/training/checkpoint164 --target-model-path ~/model_weights/model-7b 

"""
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from datetime import datetime

# model = AutoModelForCausalLM.from_pretrained(model_params['model_id'], torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
# model = PeftModel.from_pretrained(model, model_params['adapter'], device_map='auto', torch_dtype=torch.bfloat16)
# model = model.merge_and_unload()  

def save_model(checkpoint_path, target_model_path):
    print(f"Loading the checkpoint from {checkpoint_path}")
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    print(f"Saving the target model to {target_model_path}")
    model.save_pretrained(target_model_path)
    tokenizer.save_pretrained(target_model_path)

    log_path = os.path.join(target_model_path, 'merge_log.txt')
    with open(log_path, 'w') as f:
        f.write(f'--checkpoint-path : {checkpoint_path}\n')
        f.write(f'--target-model-path : {target_model_path}\n')
        f.write(f'(creation date : {datetime.now()})')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, required=True)

    args = parser.parse_args()

    save_model(args.checkpoint_path, args.target_model_path)