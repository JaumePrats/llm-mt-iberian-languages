#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
echo $CUDA_VISIBLE_DEVICES

filename_prefix=TEST_nllb_auto

timestamp=$(date +"%Y%m%d-%H.%M.%S")

python /fs/surtr0/jprats/code/llm-mt-iberian-languages/src/eval_nllb_mt.py \
    --filename_prefix $filename_prefix \
    --timestamp $timestamp \
    --batch_size 1 \
    --max_length 400 \
    /fs/surtr0/jprats/data/processed/tiny_flores/eng_Latn.dev \
    /fs/surtr0/jprats/data/processed/tiny_flores/spa_Latn.dev \
    /fs/surtr0/jprats/code/llm-mt-iberian-languages \
    facebook/nllb-200-3.3B 2> /fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/$filename_prefix'_'$timestamp.log

    # src_data
    # ref_data
    # path_prefix
    # model_id


