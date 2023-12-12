#!/bin/bash
export CUDA_VISIBLE_DEVICES=2,3,4,5
echo $CUDA_VISIBLE_DEVICES

# filename_prefix='falcon_qlora_en-es20M_ebs16_linear_lr1e-4'
filename_prefix='TEST_DS_falcon_qlora'

timestamp=$(date +"%Y%m%d-%H.%M.%S")
export WANDB_ENTITY=jaume-prats-cristia
export WANDB_PROJECT=falcon_ft_test
export WANDB_NAME=$filename_prefix'_'$timestamp
# export WANDB_NAME=falcon_qlora_en-es10k_ebs16_linear_lr1e-4_20231202-15.28.56_resumed


torchrun --nproc_per_node=4 --master_port=30010 \
    /fs/surtr0/jprats/code/llm-mt-iberian-languages/src/falcon_peft_mt_ds.py \
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
    --per_device_eval_batch_size 2 \
    --logging_steps 1 \
    --eval_steps 0.11111 \
    --save_steps 0.11111 \
    --num_train_epochs 3 \
    --bf16 \
    --learning_rate 0.0001 \
    --lr_scheduler_type linear \
    --group_by_length False \
    --deepspeed /fs/surtr0/jprats/code/llm-mt-iberian-languages/src/ds_configs/simple_config.json \
    # > /fs/surtr0/jprats/code/llm-mt-iberian-languages/results/finetune/$filename_prefix'_'$timestamp.txt \
    # 2> /fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/finetune/$filename_prefix'_'$timestamp.log &
