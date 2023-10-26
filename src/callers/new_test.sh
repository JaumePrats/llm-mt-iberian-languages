#!/bin/bash
#export CUDA_VISIBLE_DEVICES=0,1

python /fs/alvis0/jprats/code/llm-mt-iberian-languages/src/eval_llm_mt.py \
    --filename_prefix log_test \
    --batch_size 10 \
    --num_beams 5 \
    --do_sample False \
    --top_k 1 \
    --max_new_tokens 60 \
    --num_fewshot 2 \
    --template_id natural_language \
    --src_examples /fs/surtr0/jprats/data/raw/flores200_dataset/devtest/eng_Latn.devtest \
    --ref_examples /fs/surtr0/jprats/data/raw/flores200_dataset/devtest/spa_Latn.devtest \
    /fs/surtr0/jprats/data/processed/tiny_flores/eng_Latn.dev \
    /fs/surtr0/jprats/data/processed/tiny_flores/spa_Latn.dev \
    /fs/alvis0/jprats/code/llm-mt-iberian-languages \
    projecte-aina/aguila-7b

    # src_data
    # ref_data
    # path_prefix
    # model_id


