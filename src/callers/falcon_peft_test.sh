#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
echo $CUDA_VISIBLE_DEVICES

export WANDB_ENTITY=jaume-prats-cristia
export WANDB_PROJECT=falcon_ft_test
export WANDB_NAME=falcon_peft_test_1
#export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

python /fs/surtr0/jprats/code/llm-mt-iberian-languages/src/falcon_peft_test.py \
    --model_name tiiuae/falcon-7b \
    --dataset_name timdettmers/openassistant-guanaco \
