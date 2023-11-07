#!/bin/bash
export CUDA_VISIBLE_DEVICES=5,6
echo $CUDA_VISIBLE_DEVICES

filename_prefix=EVAL_falcon_flores-dev

timestamp=$(date +"%Y%m%d-%H.%M.%S")

python /fs/surtr0/jprats/code/llm-mt-iberian-languages/src/eval_llm_mt.py \
    --filename_prefix $filename_prefix \
    --timestamp $timestamp \
    --batch_size 8 \
    --num_beams 5 \
    --max_new_tokens 150 \
    --num_fewshot 1 \
    --template_id simple \
    --src_examples /fs/surtr0/jprats/data/raw/flores200_dataset/devtest/eng_Latn.devtest \
    --ref_examples /fs/surtr0/jprats/data/raw/flores200_dataset/devtest/cat_Latn.devtest \
    /fs/surtr0/jprats/data/raw/flores200_dataset/dev/eng_Latn.dev \
    /fs/surtr0/jprats/data/raw/flores200_dataset/dev/spa_Latn.dev \
    /fs/surtr0/jprats/code/llm-mt-iberian-languages \
    tiiuae/falcon-7b 2> /fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/$filename_prefix'_'$timestamp.log

    # src_data
    # ref_data
    # path_prefix
    # model_id


