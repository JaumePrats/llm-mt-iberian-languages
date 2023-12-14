#!/bin/bash
export CUDA_VISIBLE_DEVICES=6,7
echo $CUDA_VISIBLE_DEVICES

filename_prefix=AGUILA_flores-devtest
num_fewshot=5

timestamp=$(date +"%Y%m%d-%H.%M.%S")

python /fs/surtr0/jprats/code/llm-mt-iberian-languages/src/eval_llm_mt.py \
    --filename_prefix $filename_prefix \
    --timestamp $timestamp \
    --batch_size 8 \
    --num_beams 5 \
    --max_new_tokens 150 \
    --num_fewshot $num_fewshot \
    --template_id simple \
    --src_examples /fs/surtr0/jprats/data/raw/flores200_dataset/dev/eng_Latn.dev \
    --ref_examples /fs/surtr0/jprats/data/raw/flores200_dataset/dev/spa_Latn.dev \
    /fs/surtr0/jprats/data/raw/flores200_dataset/devtest/eng_Latn.devtest \
    /fs/surtr0/jprats/data/raw/flores200_dataset/devtest/spa_Latn.devtest \
    /fs/surtr0/jprats/code/llm-mt-iberian-languages \
    projecte-aina/aguila-7b 2> /fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/$filename_prefix'_'$timestamp.log

    # src_data
    # ref_data
    # path_prefix
    # model_id

    # /fs/surtr0/jprats/data/raw/flores200_dataset/dev/eng_Latn.dev \
    # /fs/surtr0/jprats/data/raw/flores200_dataset/dev/spa_Latn.dev \

