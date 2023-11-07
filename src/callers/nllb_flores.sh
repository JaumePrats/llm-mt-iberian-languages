#!/bin/bash

filename_prefix=nllb_flores

timestamp=$(date +"%Y%m%d-%H.%M.%S")

python /fs/surtr0/jprats/code/llm-mt-iberian-languages/src/eval_nllb_mt.py \
    --filename_prefix $filename_prefix \
    --timestamp $timestamp \
    --batch_size 2 \
    --max_length 400 \
    --device 7 \
    /fs/surtr0/jprats/data/raw/flores200_dataset/dev/cat_Latn.dev \
    /fs/surtr0/jprats/data/raw/flores200_dataset/dev/spa_Latn.dev \
    /fs/surtr0/jprats/code/llm-mt-iberian-languages \
    facebook/nllb-200-3.3B 2> /fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/$filename_prefix'_'$timestamp.log

    # src_data
    # ref_data
    # path_prefix
    # model_id


