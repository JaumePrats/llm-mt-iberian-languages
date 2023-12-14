#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
echo GPU:$CUDA_VISIBLE_DEVICES

model=/fs/surtr0/jprats/models/base_models/oct10/aguila-7b
filename_prefix='TEST-TIME_FFT-Aguila-10OCT'

timestamp=$(date +"%Y%m%d-%H.%M.%S")

python /fs/surtr0/jprats/code/llm-mt-iberian-languages/src/eval_llm_mt.py \
    --filename_prefix $filename_prefix \
    --timestamp $timestamp \
    --batch_size 8 \
    --num_beams 5 \
    --max_new_tokens 150 \
    --num_fewshot 0 \
    --template_id simple \
    --src_examples /fs/surtr0/jprats/data/processed/04-finetuning/tiny_flores/eng_Latn.dev \
    --ref_examples /fs/surtr0/jprats/data/processed/04-finetuning/tiny_flores/spa_Latn.dev \
    /fs/surtr0/jprats/data/processed/04-finetuning/tiny_flores/eng_Latn.dev \
    /fs/surtr0/jprats/data/processed/04-finetuning/tiny_flores/spa_Latn.dev \
    /fs/surtr0/jprats/code/llm-mt-iberian-languages \
    $model \
    # 2> /fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/mt_eval/${filename_prefix}_${timestamp}.log