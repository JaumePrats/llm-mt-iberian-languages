#!/bin/bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
echo $CUDA_VISIBLE_DEVICES

filename_prefix=aguila_ntrex

timestamp=$(date +"%Y%m%d-%H.%M.%S")

python /fs/alvis0/jprats/code/llm-mt-iberian-languages/src/eval_llm_mt.py \
    --filename_prefix $filename_prefix \
    --timestamp $timestamp \
    --batch_size 8 \
    --num_beams 5 \
    --do_sample False \
    --top_k 1 \
    --max_new_tokens 128 \
    --num_fewshot 0 \
    --template_id natural_language \
    --src_examples /fs/surtr0/jprats/data/raw/flores200_dataset/devtest/eng_Latn.devtest \
    --ref_examples /fs/surtr0/jprats/data/raw/flores200_dataset/devtest/spa_Latn.devtest \
    /fs/surtr0/jprats/data/raw/NTREX/NTREX-128/newstest2019-src.eng.txt \
    /fs/surtr0/jprats/data/raw/NTREX/NTREX-128/newstest2019-ref.spa.txt \
    /fs/alvis0/jprats/code/llm-mt-iberian-languages \
    projecte-aina/aguila-7b 2> /fs/alvis0/jprats/code/llm-mt-iberian-languages/logs/$filename_prefix'_'$timestamp.log

    # src_data
    # ref_data
    # path_prefix
    # model_id