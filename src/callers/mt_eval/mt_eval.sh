#!/bin/bash

# model=tiiuae/falcon-7b
# base_prefix=FM_falcon
model=/fs/surtr0/jprats/models/merged/falcon/qlora/falcon_qlora_en-es100k_ebs256-4x1x64_linear_lr2e-4_ep1
base_prefix=LANG_TRANF_falcon
num_fewshot=0

gpus=3

src_lang=cat
tgt_lang=eng

# ===============================
# FLORES
# params: -----
eval_set=devtest
example_set=dev
# -------------
filename_prefix=${base_prefix}_flores-$eval_set

export CUDA_VISIBLE_DEVICES=$gpus
echo GPU:$CUDA_VISIBLE_DEVICES

timestamp=$(date +"%Y%m%d-%H.%M.%S")
echo FLORES: ${src_lang}' > '${tgt_lang}

python /fs/surtr0/jprats/code/llm-mt-iberian-languages/src/eval_llm_mt.py \
    --filename_prefix $filename_prefix \
    --timestamp $timestamp \
    --batch_size 8 \
    --num_beams 5 \
    --max_new_tokens 150 \
    --num_fewshot $num_fewshot \
    --template_id simple \
    --src_examples /fs/surtr0/jprats/data/raw/flores200_dataset/${example_set}/${src_lang}_Latn.${example_set} \
    --ref_examples /fs/surtr0/jprats/data/raw/flores200_dataset/${example_set}/${tgt_lang}_Latn.${example_set} \
    /fs/surtr0/jprats/data/raw/flores200_dataset/${eval_set}/${src_lang}_Latn.${eval_set} \
    /fs/surtr0/jprats/data/raw/flores200_dataset/${eval_set}/${tgt_lang}_Latn.${eval_set} \
    /fs/surtr0/jprats/code/llm-mt-iberian-languages \
    $model 2> /fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/mt_eval/${filename_prefix}_${src_lang}-${tgt_lang}_${num_fewshot}_${timestamp}.log # &

sleep 1

# ===============================
# NTREX
# params: -----
flores_example_set=dev
# -------------
filename_prefix=${base_prefix}_ntrex

export CUDA_VISIBLE_DEVICES=$gpus
echo GPU:$CUDA_VISIBLE_DEVICES

timestamp=$(date +"%Y%m%d-%H.%M.%S")
echo NTREX: ${src_lang}' > '${tgt_lang}

python /fs/surtr0/jprats/code/llm-mt-iberian-languages/src/eval_llm_mt.py \
    --filename_prefix $filename_prefix \
    --timestamp $timestamp \
    --batch_size 8 \
    --num_beams 5 \
    --max_new_tokens 250 \
    --num_fewshot $num_fewshot \
    --template_id simple \
    --src_examples /fs/surtr0/jprats/data/raw/flores200_dataset/${flores_example_set}/${src_lang}_Latn.${flores_example_set} \
    --ref_examples /fs/surtr0/jprats/data/raw/flores200_dataset/${flores_example_set}/${tgt_lang}_Latn.${flores_example_set} \
    /fs/surtr0/jprats/data/processed/evaluation/NTREX/newstest2019-ref.${src_lang}.txt \
    /fs/surtr0/jprats/data/processed/evaluation/NTREX/newstest2019-ref.${tgt_lang}.txt \
    /fs/surtr0/jprats/code/llm-mt-iberian-languages \
    $model 2> /fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/mt_eval/${filename_prefix}_${src_lang}-${tgt_lang}_${num_fewshot}_${timestamp}.log # &

sleep 1

# ===============================
# UNPC-TEST
# params: -----
unpc_eval_set=testset
unpc_example_set=devset
# -------------
filename_prefix=${base_prefix}_unpc-$unpc_eval_set

export CUDA_VISIBLE_DEVICES=$gpus
echo GPU:$CUDA_VISIBLE_DEVICES

timestamp=$(date +"%Y%m%d-%H.%M.%S")
echo UNPC: ${src_lang}' > '${tgt_lang}

python /fs/surtr0/jprats/code/llm-mt-iberian-languages/src/eval_llm_mt.py \
    --filename_prefix $filename_prefix \
    --timestamp $timestamp \
    --batch_size 8 \
    --num_beams 5 \
    --max_new_tokens 300 \
    --num_fewshot $num_fewshot \
    --template_id simple \
    --src_examples /fs/surtr0/jprats/data/processed/evaluation/UNPC/${unpc_example_set}/UNv1.0.${unpc_example_set}.${src_lang} \
    --ref_examples /fs/surtr0/jprats/data/processed/evaluation/UNPC/${unpc_example_set}/UNv1.0.${unpc_example_set}.${tgt_lang} \
    /fs/surtr0/jprats/data/processed/evaluation/UNPC/${unpc_eval_set}/UNv1.0.${unpc_eval_set}.${src_lang} \
    /fs/surtr0/jprats/data/processed/evaluation/UNPC/${unpc_eval_set}/UNv1.0.${unpc_eval_set}.${tgt_lang} \
    /fs/surtr0/jprats/code/llm-mt-iberian-languages \
    $model 2> /fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/mt_eval/${filename_prefix}_${src_lang}-${tgt_lang}_${num_fewshot}_${timestamp}.log # &