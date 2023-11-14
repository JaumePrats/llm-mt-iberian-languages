#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
echo $CUDA_VISIBLE_DEVICES

filename_prefix=falcon_peft_mt_1.0-slowTokenizer

timestamp=$(date +"%Y%m%d-%H.%M.%S")
export WANDB_ENTITY=jaume-prats-cristia
export WANDB_PROJECT=falcon_ft_test
export WANDB_NAME=$filename_prefix'_'$timestamp

python /fs/surtr0/jprats/code/llm-mt-iberian-languages/src/falcon_peft_mt.py \
    --model_name tiiuae/falcon-7b \
    --dataset_name /fs/surtr0/jprats/data/processed/flores_text_dev_eng-spa-cat.jsonl \
    --output_dir /fs/surtr0/jprats/models/$filename_prefix'_'$timestamp \
    --max_steps 500
   2> /fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/$filename_prefix'_'$timestamp.log

# /fs/surtr0/jprats/data/processed/tinyflores_dev_eng-spa.jsonl

    # --group_by_length True \
    # --packing False \