# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field
from typing import Optional, Union
import os

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
)
from peft.tuners.lora import LoraLayer

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import wandb


########################################################################
# This is a fully working simple example to use trl's RewardTrainer.
#
# This example fine-tunes any causal language model (GPT-2, GPT-Neo, etc.)
# by using the RewardTrainer from trl, we will leverage PEFT library to finetune
# adapters on the model.
#
########################################################################


# Define and parse arguments.


@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu"})

    per_device_train_batch_size: Optional[int] = field(default=4)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[float] = field(default=2e-4)
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[int] = field(default=0.001)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=16)
    max_seq_length: Optional[int] = field(default=2048)
    model_name: Optional[str] = field(
        default="tiiuae/falcon-7b",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={
            "help": "Local path to a saved checkpoint as saved by a previous instance of Trainer. Defaults to False"
        },
    )
    dataset_name: Optional[str] = field(
        default="timdettmers/openassistant-guanaco",
        metadata={"help": "The preference dataset to use."},
    )
    dataset_files: Optional[list[str]] = field(
        default="",
        metadata={"help": "Dataset files to use for finetuning"},
    )
    train_split: Optional[str] = field(
        default='',
        metadata={"help": "Split of the training set that will be used. Syntax: [:10000], [10:20], [:10%]"},
    )
    validation_files: Optional[list[str]] = field(
        default="",
        metadata={"help": "Dataset files to use for validation"},
    )
    use_4bit: Optional[bool] = field(
        default=True,
        metadata={"help": "Activate 4bit precision base model loading"},
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables bf16 training."},
    )
    packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Use packing dataset creating."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="paged_adamw_32bit",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={"help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"},
    )
    max_steps: int = field(default=10000, metadata={"help": "How many optimizer update steps to take"})
    warmup_ratio: float = field(default=0.03, metadata={"help": "Fraction of steps to do a warmup for"})
    group_by_length: bool = field(
        default=True,
        metadata={
            "help": "Group sequences into batches with same length. Saves memory and speeds up training considerably."
        },
    )
    save_steps: float = field(default=0.1, metadata={"help": "Save checkpoint every X updates steps. Should be an integer or a float in range [0,1). If smaller than 1, will be interpreted as ratio of total training steps."})
    logging_steps: float = field(default=0.01, metadata={"help": "Log every X updates steps. Should be an integer or a float in range [0,1). If smaller than 1, will be interpreted as ratio of total training steps."})
    evaluation_strategy: str = field(
        default="no", 
        metadata={"help": "The evaluation strategy to adopt during training. Possible values are: 'no', 'steps', 'epoch'."})
    eval_steps: Optional[float] = field(
        default=None, 
        metadata={"help": "Number of update steps between two evaluations if evaluation_strategy='steps'. Float in range [0,1). If smaller than 1, will be interpreted as ratio of total training steps."})
    output_dir: Optional[str] = field(
        default="/fs/surtr0/jprats/models/first_ft_test",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

print(50*'=')
print("FINETUNING PARAMETERS:")
print('base model:', script_args.model_name)
if script_args.resume_from_checkpoint:
    print('resume_from_checkpoint:', script_args.resume_from_checkpoint)
print(50*'-')
print('train_split:', script_args.train_split)
print('dataset_files:')
for f in script_args.dataset_files:
    print(f'\t{f}')
print('validation_files:')
for f in script_args.validation_files:
    print(f'\t{f}')
print(50*'-')
print('output_dir:', script_args.output_dir)
print(50*'-')
print('learning_rate:', script_args.learning_rate)
print('lr_scheduler_type:', script_args.lr_scheduler_type)
print('effective batch size:', script_args.per_device_train_batch_size * script_args.gradient_accumulation_steps)
print('  per_device_train_batch_size:', script_args.per_device_train_batch_size)
print('  gradient_accumulation_steps:', script_args.gradient_accumulation_steps)
print('  CUDA Devices:', os.environ['CUDA_VISIBLE_DEVICES'])
print('num_train_epochs:', script_args.num_train_epochs)
print('warmup_ratio:', script_args.warmup_ratio)
print('group_by_length:', script_args.group_by_length)
print('evaluation_strategy:', script_args.evaluation_strategy)
print('eval_steps:', script_args.eval_steps)
print(50*'-')
print('lora_r:', script_args.lora_r)
print('lora_alpha:', script_args.lora_alpha)
print(50*'-')
print('bf16:', script_args.bf16)
print(50*'-')
print('use_4bit:', script_args.use_4bit)
print('bnb_4bit_quant_type:', script_args.bnb_4bit_quant_type)
print('bnb_4bit_compute_dtype:', script_args.bnb_4bit_compute_dtype)
print(50*'=')
        
class mySFTTrainer(SFTTrainer):

    def _prepare_non_packed_dataloader(
        self, tokenizer, dataset, dataset_text_field, max_seq_len, formatting_func=None
    ):
        use_formatting_func = formatting_func is not None and dataset_text_field is None
        self._dataset_sanity_checked = False

        # Inspired from: https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt
        def tokenize(element):
            # import pdb; pdb.set_trace()
            outputs = tokenizer(
                element[dataset_text_field] if not use_formatting_func else formatting_func(element),
                truncation=True,
                padding=False,
                max_length=max_seq_len,
                return_overflowing_tokens=False,
                return_length=False,
            )

            # ======================================
            # addding eos token
            for input_id, attention_mask in zip(outputs["input_ids"], outputs["attention_mask"]):
                input_id.append(tokenizer.eos_token_id)
                attention_mask.append(1)
            # ======================================

            if use_formatting_func and not self._dataset_sanity_checked:
                if not isinstance(formatting_func(element), list):
                    raise ValueError(
                        "The `formatting_func` should return a list of processed strings since it can lead to silent bugs."
                    )
                else:
                    self._dataset_sanity_checked = True

            return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}

        tokenized_dataset = dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=self.dataset_num_proc,
            batch_size=self.dataset_batch_size,
        )

        return tokenized_dataset

def create_and_prepare_model(args):
    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.use_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=args.use_nested_quant,
    )

    if compute_dtype == torch.float16 and args.use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
            print("=" * 80)

    # device_map = {"": 0}

    # model = AutoModelForCausalLM.from_pretrained(
    #     args.model_name, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
    # )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, quantization_config=bnb_config, trust_remote_code=True
    )

    peft_config = LoraConfig(
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        r=script_args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "query_key_value",
            "dense",
            "dense_h_to_4h",
            "dense_4h_to_h",
        ],  # , "word_embeddings", "lm_head"],
    )

    # using uncommon token as pad_token (pad_token must be passed as argument or it won't be saved in the config file of the tokenizer)
    if script_args.model_name == 'tiiuae/falcon-7b':
        tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, padding_side='left', trust_remote_code=True, pad_token = '~~~~~~~~')
    elif script_args.model_name == 'projecte-aina/aguila-7b':
        tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, padding_side='left', trust_remote_code=True, pad_token = '~~~')
    else: 
        tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, padding_side='left', trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token # in this case the model will not learn to predict eos_token

    return model, peft_config, tokenizer

training_arguments = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optim=script_args.optim,
    save_steps=script_args.save_steps,
    logging_steps=script_args.logging_steps,
    evaluation_strategy=script_args.evaluation_strategy,
    eval_steps=script_args.eval_steps,
    learning_rate=script_args.learning_rate,
    fp16=script_args.fp16,
    bf16=script_args.bf16,
    max_grad_norm=script_args.max_grad_norm,
    # max_steps=script_args.max_steps,
    num_train_epochs=script_args.num_train_epochs,
    warmup_ratio=script_args.warmup_ratio,
    group_by_length=script_args.group_by_length,
    lr_scheduler_type=script_args.lr_scheduler_type,
    gradient_checkpointing=script_args.gradient_checkpointing
)

model, peft_config, tokenizer = create_and_prepare_model(script_args)
model.config.use_cache = False
dataset = load_dataset('json', data_files={'train': script_args.dataset_files}, split=f'train{script_args.train_split}')
print("Resulting dataset:")
print(dataset)
valid_dataset = load_dataset('json', data_files={'validation': script_args.validation_files}, split='validation')
print("Resulting validation dataset:")
print(valid_dataset)


# instruction_template = "###SRC"
# response_template = "###TGT"
response_template = "\n"
collator = DataCollatorForCompletionOnlyLM(
    # instruction_template=instruction_template, 
    response_template=response_template, 
    tokenizer=tokenizer, 
    mlm=False
    )

print(dataset)
print(script_args.packing)
print(script_args.group_by_length)

trainer = mySFTTrainer(
    model=model,
    train_dataset=dataset,
    eval_dataset=valid_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    data_collator=collator,
    max_seq_length=script_args.max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=script_args.packing,
)


for name, module in trainer.model.named_modules():
    if isinstance(module, LoraLayer):
        if script_args.bf16:
            module = module.to(torch.bfloat16)
    if "norm" in name:
        module = module.to(torch.float32)
    if "lm_head" in name or "embed_tokens" in name:
        if hasattr(module, "weight"):
            if script_args.bf16 and module.weight.dtype == torch.float32:
                module = module.to(torch.bfloat16)

if script_args.resume_from_checkpoint == None:
    resume = False
else: 
    resume = script_args.resume_from_checkpoint

# wandb.init(id='5hbbcgit')
trainer.train(resume_from_checkpoint=resume)