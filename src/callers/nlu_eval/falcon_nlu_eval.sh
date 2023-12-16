#!/bin/bash

prefix=TEST_XQUAD_5SHOT
model=tiiuae/falcon-7b
tasks=xquad_es
num_fewshot=5
device=0


# GETTING MODEL NAME:
model_name=$model
# Set the Internal Field Separator (IFS) to space temporarily
IFS='/'
read -ra string_array <<< "$model_name"
# Reset IFS to its default value (whitespace)
IFS=$' \t\n'
array_length=${#string_array[@]}
# Calculate the index of the last element
last_index=$((array_length - 1))
# Access the last element
model_name="${string_array[last_index]}"

timestamp=$(date +"%Y%m%d-%H.%M.%S")

echo "---------------------------------------------------"
echo NLU EVAL: ${prefix}_${tasks//,/-}_nshot${num_fewshot}_${timestamp}
echo "---------------------------------------------------"
echo "model: $model"
echo "tasks: $tasks"
echo "num fewhot: $num_fewshot"
echo "cuda device: $device"
echo "---------------------------------------------------"

python /fs/surtr0/jprats/code/lm-evaluation-harness/main.py \
    --model hf-causal \
    --model_args pretrained=$model,dtype=float16,trust_remote_code=True \
    --tasks $tasks \
    --num_fewshot $num_fewshot \
    --device cuda:${device} \
    # > /fs/surtr0/jprats/code/llm-mt-iberian-languages/results/nlu_eval/${prefix}_${tasks//,/-}_${model_name}_nshot${num_fewshot}_${timestamp} \
    # 2> /fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/nlu_eval/${prefix}_${tasks//,/-}_${model_name}_nshot${num_fewshot}_${timestamp}