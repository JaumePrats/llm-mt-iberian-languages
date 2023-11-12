#!/bin/bash

num_fewshot=3
# model=tiiuae/falcon-7b
model=projecte-aina/aguila-7b
filename_prefix=EVAL_aguila_flores-devtest

# ===============================

export CUDA_VISIBLE_DEVICES=2
echo GPU:$CUDA_VISIBLE_DEVICES

src_lang=eng
tgt_lang=spa

timestamp=$(date +"%Y%m%d-%H.%M.%S")
echo $src_lang'>'$tgt_lang

python /fs/surtr0/jprats/code/llm-mt-iberian-languages/src/eval_llm_mt.py \
    --filename_prefix $filename_prefix \
    --timestamp $timestamp \
    --batch_size 8 \
    --num_beams 5 \
    --max_new_tokens 150 \
    --num_fewshot $num_fewshot \
    --template_id simple \
    --src_examples /fs/surtr0/jprats/data/raw/flores200_dataset/dev/$src_lang'_Latn.dev' \
    --ref_examples /fs/surtr0/jprats/data/raw/flores200_dataset/dev/$tgt_lang'_Latn.dev' \
    /fs/surtr0/jprats/data/raw/flores200_dataset/devtest/$src_lang'_Latn.devtest' \
    /fs/surtr0/jprats/data/raw/flores200_dataset/devtest/$tgt_lang'_Latn.devtest' \
    /fs/surtr0/jprats/code/llm-mt-iberian-languages \
    $model 2> /fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/$filename_prefix'_'$src_lang'-'$tgt_lang'_'$timestamp.log &

# ===============================

export CUDA_VISIBLE_DEVICES=3
echo GPU:$CUDA_VISIBLE_DEVICES

src_lang=spa
tgt_lang=eng

timestamp=$(date +"%Y%m%d-%H.%M.%S")
echo $src_lang'>'$tgt_lang

python /fs/surtr0/jprats/code/llm-mt-iberian-languages/src/eval_llm_mt.py \
    --filename_prefix $filename_prefix \
    --timestamp $timestamp \
    --batch_size 8 \
    --num_beams 5 \
    --max_new_tokens 150 \
    --num_fewshot $num_fewshot \
    --template_id simple \
    --src_examples /fs/surtr0/jprats/data/raw/flores200_dataset/dev/$src_lang'_Latn.dev' \
    --ref_examples /fs/surtr0/jprats/data/raw/flores200_dataset/dev/$tgt_lang'_Latn.dev' \
    /fs/surtr0/jprats/data/raw/flores200_dataset/devtest/$src_lang'_Latn.devtest' \
    /fs/surtr0/jprats/data/raw/flores200_dataset/devtest/$tgt_lang'_Latn.devtest' \
    /fs/surtr0/jprats/code/llm-mt-iberian-languages \
    $model 2> /fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/$filename_prefix'_'$src_lang'-'$tgt_lang'_'$timestamp.log &

# ===============================

export CUDA_VISIBLE_DEVICES=4
echo GPU:$CUDA_VISIBLE_DEVICES

src_lang=eng
tgt_lang=cat

timestamp=$(date +"%Y%m%d-%H.%M.%S")
echo $src_lang'>'$tgt_lang

python /fs/surtr0/jprats/code/llm-mt-iberian-languages/src/eval_llm_mt.py \
    --filename_prefix $filename_prefix \
    --timestamp $timestamp \
    --batch_size 8 \
    --num_beams 5 \
    --max_new_tokens 150 \
    --num_fewshot $num_fewshot \
    --template_id simple \
    --src_examples /fs/surtr0/jprats/data/raw/flores200_dataset/dev/$src_lang'_Latn.dev' \
    --ref_examples /fs/surtr0/jprats/data/raw/flores200_dataset/dev/$tgt_lang'_Latn.dev' \
    /fs/surtr0/jprats/data/raw/flores200_dataset/devtest/$src_lang'_Latn.devtest' \
    /fs/surtr0/jprats/data/raw/flores200_dataset/devtest/$tgt_lang'_Latn.devtest' \
    /fs/surtr0/jprats/code/llm-mt-iberian-languages \
    $model 2> /fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/$filename_prefix'_'$src_lang'-'$tgt_lang'_'$timestamp.log &

# ===============================

export CUDA_VISIBLE_DEVICES=5
echo GPU:$CUDA_VISIBLE_DEVICES

src_lang=cat
tgt_lang=eng

timestamp=$(date +"%Y%m%d-%H.%M.%S")
echo $src_lang'>'$tgt_lang

python /fs/surtr0/jprats/code/llm-mt-iberian-languages/src/eval_llm_mt.py \
    --filename_prefix $filename_prefix \
    --timestamp $timestamp \
    --batch_size 8 \
    --num_beams 5 \
    --max_new_tokens 150 \
    --num_fewshot $num_fewshot \
    --template_id simple \
    --src_examples /fs/surtr0/jprats/data/raw/flores200_dataset/dev/$src_lang'_Latn.dev' \
    --ref_examples /fs/surtr0/jprats/data/raw/flores200_dataset/dev/$tgt_lang'_Latn.dev' \
    /fs/surtr0/jprats/data/raw/flores200_dataset/devtest/$src_lang'_Latn.devtest' \
    /fs/surtr0/jprats/data/raw/flores200_dataset/devtest/$tgt_lang'_Latn.devtest' \
    /fs/surtr0/jprats/code/llm-mt-iberian-languages \
    $model 2> /fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/$filename_prefix'_'$src_lang'-'$tgt_lang'_'$timestamp.log &

# ===============================

export CUDA_VISIBLE_DEVICES=6
echo GPU:$CUDA_VISIBLE_DEVICES

src_lang=cat
tgt_lang=spa

timestamp=$(date +"%Y%m%d-%H.%M.%S")
echo $src_lang'-'$tgt_lang

python /fs/surtr0/jprats/code/llm-mt-iberian-languages/src/eval_llm_mt.py \
    --filename_prefix $filename_prefix \
    --timestamp $timestamp \
    --batch_size 8 \
    --num_beams 5 \
    --max_new_tokens 150 \
    --num_fewshot $num_fewshot \
    --template_id simple \
    --src_examples /fs/surtr0/jprats/data/raw/flores200_dataset/dev/$src_lang'_Latn.dev' \
    --ref_examples /fs/surtr0/jprats/data/raw/flores200_dataset/dev/$tgt_lang'_Latn.dev' \
    /fs/surtr0/jprats/data/raw/flores200_dataset/devtest/$src_lang'_Latn.devtest' \
    /fs/surtr0/jprats/data/raw/flores200_dataset/devtest/$tgt_lang'_Latn.devtest' \
    /fs/surtr0/jprats/code/llm-mt-iberian-languages \
    $model 2> /fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/$filename_prefix'_'$src_lang'-'$tgt_lang'_'$timestamp.log &

# ===============================

export CUDA_VISIBLE_DEVICES=7
echo GPU:$CUDA_VISIBLE_DEVICES

src_lang=spa
tgt_lang=cat

timestamp=$(date +"%Y%m%d-%H.%M.%S")
echo $src_lang'-'$tgt_lang

python /fs/surtr0/jprats/code/llm-mt-iberian-languages/src/eval_llm_mt.py \
    --filename_prefix $filename_prefix \
    --timestamp $timestamp \
    --batch_size 8 \
    --num_beams 5 \
    --max_new_tokens 150 \
    --num_fewshot $num_fewshot \
    --template_id simple \
    --src_examples /fs/surtr0/jprats/data/raw/flores200_dataset/dev/$src_lang'_Latn.dev' \
    --ref_examples /fs/surtr0/jprats/data/raw/flores200_dataset/dev/$tgt_lang'_Latn.dev' \
    /fs/surtr0/jprats/data/raw/flores200_dataset/devtest/$src_lang'_Latn.devtest' \
    /fs/surtr0/jprats/data/raw/flores200_dataset/devtest/$tgt_lang'_Latn.devtest' \
    /fs/surtr0/jprats/code/llm-mt-iberian-languages \
    $model 2> /fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/$filename_prefix'_'$src_lang'-'$tgt_lang'_'$timestamp.log &

# ===============================
