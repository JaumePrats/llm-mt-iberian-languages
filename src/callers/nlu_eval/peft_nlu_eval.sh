#!/bin/bash

prefix=TEST_adapter
model=tiiuae/falcon-7b
tasks=xnli_ca
num_fewshot=0
device=5

echo "NLU EVAL --------------------------------"
echo "model: $model"
echo "tasks: $tasks"
echo "num fewhot: $model"
echo "cuda device: $device"
echo "-----------------------------------------"

timestamp=$(date +"%Y%m%d-%H.%M.%S")

echo 
python /fs/surtr0/jprats/code/lm-evaluation-harness/main.py \
    --model hf-causal \
    --model_args pretrained=$model,dtype=torch.bfloat16,trust_remote_code=True \
    --tasks $tasks \
    --num_fewshot 0 \
    --device cuda:${device} \
    > /fs/surtr0/jprats/code/llm-mt-iberian-languages/results/nlu_eval/${prefix}_"${tasks//,/-}"_nshot${num_fewshot}_${timestamp} \
    # 2> /fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/nlu_eval/${prefix}_"${tasks//,/-}"_nshot${num_fewshot}_${timestamp}