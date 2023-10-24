#!/bin/bash

python /fs/alvis0/jprats/code/llm-mt-iberian-languages/eval_llm_mt.py \
    --filename_prefix first_code_test \
    --num_beams 5 \
    --do_sample False \
    --top_k 50 \
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


