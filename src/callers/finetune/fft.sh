#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo $CUDA_VISIBLE_DEVICES

filename_prefix='falcon_fft_en-es10k_ebs256-4-1x8x32_linear_lr2e-5'

timestamp=$(date +"%Y%m%d-%H.%M.%S")
export WANDB_ENTITY=jaume-prats-cristia
export WANDB_PROJECT=falcon_ft_test
export WANDB_NAME=$filename_prefix'_'$timestamp

    # --bf16 \
    # --per_device_train_batch_size 1 \
    # --gradient_accumulation_steps 64 \


python /fs/surtr0/jprats/code/llm-mt-iberian-languages/src/fft.py \
    --model_name tiiuae/falcon-7b \
    --dataset_files \
    '/fs/surtr0/jprats/data/processed/04-finetuning/en-es_europarl-unpc/europarl-unpc_en-es_bidir.jsonl' \
    --train_split '[:20000]' \
    --validation_files \
    '/fs/surtr0/jprats/data/processed/04-finetuning/devsets/flores_dev_eng-spa.jsonl' \
    '/fs/surtr0/jprats/data/processed/04-finetuning/devsets/flores_dev_spa-eng.jsonl' \
    '/fs/surtr0/jprats/data/processed/04-finetuning/devsets/unpc_dev_en-es_unidir.jsonl' \
    '/fs/surtr0/jprats/data/processed/04-finetuning/devsets/unpc_dev_es-en_unidir.jsonl' \
    --output_dir /fs/surtr0/jprats/models/checkpoints/$filename_prefix'_'$timestamp \
    --evaluation_strategy steps \
    --per_device_eval_batch_size 8 \
    --bf16 \
    --logging_steps 1 \
    --eval_steps 0.05555 \
    --save_steps 0.11111 \
    --num_train_epochs 3 \
    --learning_rate 0.00002 \
    --lr_scheduler_type linear \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 32 \
    --group_by_length False \
    --gradient_checkpointing False \
    > /fs/surtr0/jprats/code/llm-mt-iberian-languages/results/finetune/$filename_prefix'_'$timestamp.txt \
    2> /fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/finetune/$filename_prefix'_'$timestamp.log &

