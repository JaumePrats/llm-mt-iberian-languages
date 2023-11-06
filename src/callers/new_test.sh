#!/bin/bash
export CUDA_VISIBLE_DEVICES=6,7
echo $CUDA_VISIBLE_DEVICES

filename_prefix=TEST_infer-env

timestamp=$(date +"%Y%m%d-%H.%M.%S")

python /fs/surtr0/jprats/code/llm-mt-iberian-languages/src/eval_llm_mt.py \
    --filename_prefix $filename_prefix \
    --timestamp $timestamp \
    --batch_size 8 \
    --num_beams 1 \
    --max_new_tokens 60 \
    --num_fewshot 0 \
    --template_id simple \
    --src_examples /fs/surtr0/jprats/data/raw/flores200_dataset/devtest/eng_Latn.devtest \
    --ref_examples /fs/surtr0/jprats/data/raw/flores200_dataset/devtest/spa_Latn.devtest \
    /fs/surtr0/jprats/data/processed/tiny_flores/eng_Latn.dev \
    /fs/surtr0/jprats/data/processed/tiny_flores/spa_Latn.dev \
    /fs/surtr0/jprats/code/llm-mt-iberian-languages \
    projecte-aina/aguila-7b 2> /fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/$filename_prefix'_'$timestamp.log

    # src_data
    # ref_data
    # path_prefix
    # model_id


