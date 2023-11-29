#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
echo $CUDA_VISIBLE_DEVICES

filename_prefix='TEST_falcon_lora'

timestamp=$(date +"%Y%m%d-%H.%M.%S")
export WANDB_ENTITY=jaume-prats-cristia
export WANDB_PROJECT=falcon_ft_test
export WANDB_NAME=$filename_prefix'_'$timestamp

python /fs/surtr0/jprats/code/llm-mt-iberian-languages/src/falcon_lora_mt.py \
    --model_name /fs/surtr0/jprats/models/base_models/falcon-7b \
    --dataset_files \
    '/fs/surtr0/jprats/data/processed/finetuning/europarl/europarl-clean_en-es_bidir.jsonl' \
    --train_split '[:10000]' \
    --validation_files \
    '/fs/surtr0/jprats/data/processed/finetuning/parallel_ft/valid/flores_dev_eng-spa.jsonl' \
    --output_dir /fs/surtr0/jprats/models/$filename_prefix'_'$timestamp \
    --evaluation_strategy steps \
    --eval_steps 50 \
    --max_steps 10000 \
    --bf16 \
    --learning_rate 0.00002 \
    --lr_scheduler_type linear \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    # 2> /fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/finetune/$filename_prefix'_'$timestamp.log

# Different directory per run:
    # --output_dir /fs/surtr0/jprats/models/$filename_prefix'_'$timestamp \


# optional arguments:
#   -h, --help            show this help message and exit
#   --local_rank LOCAL_RANK
#                         Used for multi-gpu (default: -1)
#   --per_device_train_batch_size PER_DEVICE_TRAIN_BATCH_SIZE
#   --per_device_eval_batch_size PER_DEVICE_EVAL_BATCH_SIZE
#   --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
#   --learning_rate LEARNING_RATE
#   --max_grad_norm MAX_GRAD_NORM
#   --weight_decay WEIGHT_DECAY
#   --lora_alpha LORA_ALPHA
#   --lora_dropout LORA_DROPOUT
#   --lora_r LORA_R
#   --max_seq_length MAX_SEQ_LENGTH
#   --model_name MODEL_NAME
#                         The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc. (default: tiiuae/falcon-7b)
#   --dataset_name DATASET_NAME
#                         The preference dataset to use. (default: timdettmers/openassistant-guanaco)
#   --dataset_files DATASET_FILES [DATASET_FILES ...]
#                         Dataset files to use for finetuning (default: None)
#   --use_4bit [USE_4BIT]
#                         Activate 4bit precision base model loading (default: True)
#   --no_use_4bit         Activate 4bit precision base model loading (default: False)
#   --use_nested_quant [USE_NESTED_QUANT]
#                         Activate nested quantization for 4bit base models (default: False)
#   --bnb_4bit_compute_dtype BNB_4BIT_COMPUTE_DTYPE
#                         Compute dtype for 4bit base models (default: float16)
#   --bnb_4bit_quant_type BNB_4BIT_QUANT_TYPE
#                         Quantization type fp4 or nf4 (default: nf4)
#   --num_train_epochs NUM_TRAIN_EPOCHS
#                         The number of training epochs for the reward model. (default: 1)
#   --fp16 [FP16]         Enables fp16 training. (default: False)
#   --bf16 [BF16]         Enables bf16 training. (default: False)
#   --packing [PACKING]   Use packing dataset creating. (default: False)
#   --gradient_checkpointing [GRADIENT_CHECKPOINTING]
#                         Enables gradient checkpointing. (default: True)
#   --no_gradient_checkpointing
#                         Enables gradient checkpointing. (default: False)
#   --optim OPTIM         The optimizer to use. (default: paged_adamw_32bit)
#   --lr_scheduler_type LR_SCHEDULER_TYPE
#                         Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis (default: constant)
#   --max_steps MAX_STEPS
#                         How many optimizer update steps to take (default: 10000)
#   --warmup_ratio WARMUP_RATIO
#                         Fraction of steps to do a warmup for (default: 0.03)
#   --group_by_length [GROUP_BY_LENGTH]
#                         Group sequences into batches with same length. Saves memory and speeds up training considerably. (default: True)
#   --no_group_by_length  Group sequences into batches with same length. Saves memory and speeds up training considerably. (default: False)
#   --save_steps SAVE_STEPS
#                         Save checkpoint every X updates steps. (default: 10)
#   --logging_steps LOGGING_STEPS
#                         Log every X updates steps. (default: 10)
#   --output_dir OUTPUT_DIR
#                         The output directory where the model predictions and checkpoints will be written. (default: /fs/surtr0/jprats/models/first_ft_test)
