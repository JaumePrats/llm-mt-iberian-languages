#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
echo GPU:$CUDA_VISIBLE_DEVICES

filename_prefix=EVAL_falcon_flores-dev
src_lang=spa
tgt_lang=cat

timestamp=$(date +"%Y%m%d-%H.%M.%S")
echo $src_lang'>'$tgt_lang

python /fs/surtr0/jprats/code/llm-mt-iberian-languages/src/eval_llm_mt.py \
    --filename_prefix $filename_prefix \
    --timestamp $timestamp \
    --batch_size 8 \
    --num_beams 5 \
    --max_new_tokens 150 \
    --num_fewshot 8 \
    --template_id simple \
    --src_examples /fs/surtr0/jprats/data/raw/flores200_dataset/devtest/$src_lang'_Latn.devtest' \
    --ref_examples /fs/surtr0/jprats/data/raw/flores200_dataset/devtest/$tgt_lang'_Latn.devtest' \
    /fs/surtr0/jprats/data/raw/flores200_dataset/dev/$src_lang'_Latn.dev' \
    /fs/surtr0/jprats/data/raw/flores200_dataset/dev/$tgt_lang'_Latn.dev' \
    /fs/surtr0/jprats/code/llm-mt-iberian-languages \
    tiiuae/falcon-7b 2> /fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/$filename_prefix'_'$timestamp.log

    # src_data
    # ref_data
    # path_prefix
    # model_id


    # --src_examples /fs/surtr0/jprats/data/raw/flores200_dataset/devtest/$src_lang'_Latn.devtest' \
    # --ref_examples /fs/surtr0/jprats/data/raw/flores200_dataset/devtest/$tgt_lang'_Latn.devtest' \

