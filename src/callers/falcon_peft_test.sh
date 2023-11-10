#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
echo $CUDA_VISIBLE_DEVICES

filename_prefix=falcon_peft_test_2

timestamp=$(date +"%Y%m%d-%H.%M.%S")
export WANDB_ENTITY=jaume-prats-cristia
export WANDB_PROJECT=falcon_ft_test
export WANDB_NAME=$filename_prefix'_'$timestamp

python /fs/surtr0/jprats/code/llm-mt-iberian-languages/src/falcon_peft_test.py \
    --model_name tiiuae/falcon-7b \
    --dataset_name timdettmers/openassistant-guanaco \
    2> /fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/$filename_prefix'_'$timestamp.log
