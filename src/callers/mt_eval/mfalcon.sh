#!/bin/bash

num_fewshot=5
model=tiiuae/falcon-7b
eval_set=devtest
example_set=dev
filename_prefix=FM_falcon_flores-$eval_set
gpus=(5 6)


# ===============================

# export CUDA_VISIBLE_DEVICES=${gpus[0]},${gpus[1]}
# echo GPU:$CUDA_VISIBLE_DEVICES

# src_lang=cat
# tgt_lang=eng

# timestamp=$(date +"%Y%m%d-%H.%M.%S")
# echo ${src_lang}' > '${tgt_lang}

# python /fs/surtr0/jprats/code/llm-mt-iberian-languages/src/eval_llm_mt.py \
#     --filename_prefix $filename_prefix \
#     --timestamp $timestamp \
#     --batch_size 8 \
#     --num_beams 5 \
#     --max_new_tokens 150 \
#     --num_fewshot $num_fewshot \
#     --template_id simple \
#     --src_examples /fs/surtr0/jprats/data/raw/flores200_dataset/${example_set}/${src_lang}_Latn.${example_set} \
#     --ref_examples /fs/surtr0/jprats/data/raw/flores200_dataset/${example_set}/${tgt_lang}_Latn.${example_set} \
#     /fs/surtr0/jprats/data/raw/flores200_dataset/${eval_set}/${src_lang}_Latn.${eval_set} \
#     /fs/surtr0/jprats/data/raw/flores200_dataset/${eval_set}/${tgt_lang}_Latn.${eval_set} \
#     /fs/surtr0/jprats/code/llm-mt-iberian-languages \
#     $model 2> /fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/mt_eval/${filename_prefix}_${src_lang}-${tgt_lang}_${num_fewshot}_${timestamp}.log # &

# # ===============================
# sleep 1

# export CUDA_VISIBLE_DEVICES=${gpus[0]},${gpus[1]}
# echo GPU:$CUDA_VISIBLE_DEVICES

# src_lang=cat
# tgt_lang=spa

# timestamp=$(date +"%Y%m%d-%H.%M.%S")
# echo ${src_lang}' > '${tgt_lang}

# python /fs/surtr0/jprats/code/llm-mt-iberian-languages/src/eval_llm_mt.py \
#     --filename_prefix $filename_prefix \
#     --timestamp $timestamp \
#     --batch_size 8 \
#     --num_beams 5 \
#     --max_new_tokens 150 \
#     --num_fewshot $num_fewshot \
#     --template_id simple \
#     --src_examples /fs/surtr0/jprats/data/raw/flores200_dataset/${example_set}/${src_lang}_Latn.${example_set} \
#     --ref_examples /fs/surtr0/jprats/data/raw/flores200_dataset/${example_set}/${tgt_lang}_Latn.${example_set} \
#     /fs/surtr0/jprats/data/raw/flores200_dataset/${eval_set}/${src_lang}_Latn.${eval_set} \
#     /fs/surtr0/jprats/data/raw/flores200_dataset/${eval_set}/${tgt_lang}_Latn.${eval_set} \
#     /fs/surtr0/jprats/code/llm-mt-iberian-languages \
#     $model 2> /fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/mt_eval/${filename_prefix}_${src_lang}-${tgt_lang}_${num_fewshot}_${timestamp}.log # &

# # ===============================
# sleep 1

# export CUDA_VISIBLE_DEVICES=${gpus[0]},${gpus[1]}
# echo GPU:$CUDA_VISIBLE_DEVICES

# src_lang=eng
# tgt_lang=cat

# timestamp=$(date +"%Y%m%d-%H.%M.%S")
# echo ${src_lang}' > '${tgt_lang}

# python /fs/surtr0/jprats/code/llm-mt-iberian-languages/src/eval_llm_mt.py \
#     --filename_prefix $filename_prefix \
#     --timestamp $timestamp \
#     --batch_size 8 \
#     --num_beams 5 \
#     --max_new_tokens 150 \
#     --num_fewshot $num_fewshot \
#     --template_id simple \
#     --src_examples /fs/surtr0/jprats/data/raw/flores200_dataset/${example_set}/${src_lang}_Latn.${example_set} \
#     --ref_examples /fs/surtr0/jprats/data/raw/flores200_dataset/${example_set}/${tgt_lang}_Latn.${example_set} \
#     /fs/surtr0/jprats/data/raw/flores200_dataset/${eval_set}/${src_lang}_Latn.${eval_set} \
#     /fs/surtr0/jprats/data/raw/flores200_dataset/${eval_set}/${tgt_lang}_Latn.${eval_set} \
#     /fs/surtr0/jprats/code/llm-mt-iberian-languages \
#     $model 2> /fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/mt_eval/${filename_prefix}_${src_lang}-${tgt_lang}_${num_fewshot}_${timestamp}.log # &

# # ===============================
# sleep 1

# export CUDA_VISIBLE_DEVICES=${gpus[0]},${gpus[1]}
# echo GPU:$CUDA_VISIBLE_DEVICES

# src_lang=eng
# tgt_lang=spa

# timestamp=$(date +"%Y%m%d-%H.%M.%S")
# echo ${src_lang}' > '${tgt_lang}

# python /fs/surtr0/jprats/code/llm-mt-iberian-languages/src/eval_llm_mt.py \
#     --filename_prefix $filename_prefix \
#     --timestamp $timestamp \
#     --batch_size 8 \
#     --num_beams 5 \
#     --max_new_tokens 150 \
#     --num_fewshot $num_fewshot \
#     --template_id simple \
#     --src_examples /fs/surtr0/jprats/data/raw/flores200_dataset/${example_set}/${src_lang}_Latn.${example_set} \
#     --ref_examples /fs/surtr0/jprats/data/raw/flores200_dataset/${example_set}/${tgt_lang}_Latn.${example_set} \
#     /fs/surtr0/jprats/data/raw/flores200_dataset/${eval_set}/${src_lang}_Latn.${eval_set} \
#     /fs/surtr0/jprats/data/raw/flores200_dataset/${eval_set}/${tgt_lang}_Latn.${eval_set} \
#     /fs/surtr0/jprats/code/llm-mt-iberian-languages \
#     $model 2> /fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/mt_eval/${filename_prefix}_${src_lang}-${tgt_lang}_${num_fewshot}_${timestamp}.log # &

# # ===============================
# sleep 1

# export CUDA_VISIBLE_DEVICES=${gpus[0]},${gpus[1]}
# echo GPU:$CUDA_VISIBLE_DEVICES

# src_lang=spa
# tgt_lang=cat

# timestamp=$(date +"%Y%m%d-%H.%M.%S")
# echo ${src_lang}' > '${tgt_lang}

# python /fs/surtr0/jprats/code/llm-mt-iberian-languages/src/eval_llm_mt.py \
#     --filename_prefix $filename_prefix \
#     --timestamp $timestamp \
#     --batch_size 8 \
#     --num_beams 5 \
#     --max_new_tokens 150 \
#     --num_fewshot $num_fewshot \
#     --template_id simple \
#     --src_examples /fs/surtr0/jprats/data/raw/flores200_dataset/${example_set}/${src_lang}_Latn.${example_set} \
#     --ref_examples /fs/surtr0/jprats/data/raw/flores200_dataset/${example_set}/${tgt_lang}_Latn.${example_set} \
#     /fs/surtr0/jprats/data/raw/flores200_dataset/${eval_set}/${src_lang}_Latn.${eval_set} \
#     /fs/surtr0/jprats/data/raw/flores200_dataset/${eval_set}/${tgt_lang}_Latn.${eval_set} \
#     /fs/surtr0/jprats/code/llm-mt-iberian-languages \
#     $model 2> /fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/mt_eval/${filename_prefix}_${src_lang}-${tgt_lang}_${num_fewshot}_${timestamp}.log # &

# # ===============================
# sleep 1

# export CUDA_VISIBLE_DEVICES=${gpus[0]},${gpus[1]}
# echo GPU:$CUDA_VISIBLE_DEVICES

# src_lang=spa
# tgt_lang=eng

# timestamp=$(date +"%Y%m%d-%H.%M.%S")
# echo ${src_lang}' > '${tgt_lang}

# python /fs/surtr0/jprats/code/llm-mt-iberian-languages/src/eval_llm_mt.py \
#     --filename_prefix $filename_prefix \
#     --timestamp $timestamp \
#     --batch_size 8 \
#     --num_beams 5 \
#     --max_new_tokens 150 \
#     --num_fewshot $num_fewshot \
#     --template_id simple \
#     --src_examples /fs/surtr0/jprats/data/raw/flores200_dataset/${example_set}/${src_lang}_Latn.${example_set} \
#     --ref_examples /fs/surtr0/jprats/data/raw/flores200_dataset/${example_set}/${tgt_lang}_Latn.${example_set} \
#     /fs/surtr0/jprats/data/raw/flores200_dataset/${eval_set}/${src_lang}_Latn.${eval_set} \
#     /fs/surtr0/jprats/data/raw/flores200_dataset/${eval_set}/${tgt_lang}_Latn.${eval_set} \
#     /fs/surtr0/jprats/code/llm-mt-iberian-languages \
#     $model 2> /fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/mt_eval/${filename_prefix}_${src_lang}-${tgt_lang}_${num_fewshot}_${timestamp}.log # &

# # ===============================
# num_fewshot=0
# # ===============================

# export CUDA_VISIBLE_DEVICES=${gpus[1]}
# echo GPU:$CUDA_VISIBLE_DEVICES

# src_lang=cat
# tgt_lang=eng

# timestamp=$(date +"%Y%m%d-%H.%M.%S")
# echo ${src_lang}' > '${tgt_lang}

# python /fs/surtr0/jprats/code/llm-mt-iberian-languages/src/eval_llm_mt.py \
#     --filename_prefix $filename_prefix \
#     --timestamp $timestamp \
#     --batch_size 8 \
#     --num_beams 5 \
#     --max_new_tokens 150 \
#     --num_fewshot $num_fewshot \
#     --template_id simple \
#     --src_examples /fs/surtr0/jprats/data/raw/flores200_dataset/${example_set}/${src_lang}_Latn.${example_set} \
#     --ref_examples /fs/surtr0/jprats/data/raw/flores200_dataset/${example_set}/${tgt_lang}_Latn.${example_set} \
#     /fs/surtr0/jprats/data/raw/flores200_dataset/${eval_set}/${src_lang}_Latn.${eval_set} \
#     /fs/surtr0/jprats/data/raw/flores200_dataset/${eval_set}/${tgt_lang}_Latn.${eval_set} \
#     /fs/surtr0/jprats/code/llm-mt-iberian-languages \
#     $model 2> /fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/mt_eval/${filename_prefix}_${src_lang}-${tgt_lang}_${num_fewshot}_${timestamp}.log # &

# # ===============================
# sleep 1

# export CUDA_VISIBLE_DEVICES=${gpus[1]}
# echo GPU:$CUDA_VISIBLE_DEVICES

# src_lang=cat
# tgt_lang=spa

# timestamp=$(date +"%Y%m%d-%H.%M.%S")
# echo ${src_lang}' > '${tgt_lang}

# python /fs/surtr0/jprats/code/llm-mt-iberian-languages/src/eval_llm_mt.py \
#     --filename_prefix $filename_prefix \
#     --timestamp $timestamp \
#     --batch_size 8 \
#     --num_beams 5 \
#     --max_new_tokens 150 \
#     --num_fewshot $num_fewshot \
#     --template_id simple \
#     --src_examples /fs/surtr0/jprats/data/raw/flores200_dataset/${example_set}/${src_lang}_Latn.${example_set} \
#     --ref_examples /fs/surtr0/jprats/data/raw/flores200_dataset/${example_set}/${tgt_lang}_Latn.${example_set} \
#     /fs/surtr0/jprats/data/raw/flores200_dataset/${eval_set}/${src_lang}_Latn.${eval_set} \
#     /fs/surtr0/jprats/data/raw/flores200_dataset/${eval_set}/${tgt_lang}_Latn.${eval_set} \
#     /fs/surtr0/jprats/code/llm-mt-iberian-languages \
#     $model 2> /fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/mt_eval/${filename_prefix}_${src_lang}-${tgt_lang}_${num_fewshot}_${timestamp}.log # &

# # ===============================
# sleep 1

# export CUDA_VISIBLE_DEVICES=${gpus[1]}
# echo GPU:$CUDA_VISIBLE_DEVICES

# src_lang=eng
# tgt_lang=cat

# timestamp=$(date +"%Y%m%d-%H.%M.%S")
# echo ${src_lang}' > '${tgt_lang}

# python /fs/surtr0/jprats/code/llm-mt-iberian-languages/src/eval_llm_mt.py \
#     --filename_prefix $filename_prefix \
#     --timestamp $timestamp \
#     --batch_size 8 \
#     --num_beams 5 \
#     --max_new_tokens 150 \
#     --num_fewshot $num_fewshot \
#     --template_id simple \
#     --src_examples /fs/surtr0/jprats/data/raw/flores200_dataset/${example_set}/${src_lang}_Latn.${example_set} \
#     --ref_examples /fs/surtr0/jprats/data/raw/flores200_dataset/${example_set}/${tgt_lang}_Latn.${example_set} \
#     /fs/surtr0/jprats/data/raw/flores200_dataset/${eval_set}/${src_lang}_Latn.${eval_set} \
#     /fs/surtr0/jprats/data/raw/flores200_dataset/${eval_set}/${tgt_lang}_Latn.${eval_set} \
#     /fs/surtr0/jprats/code/llm-mt-iberian-languages \
#     $model 2> /fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/mt_eval/${filename_prefix}_${src_lang}-${tgt_lang}_${num_fewshot}_${timestamp}.log # &

# # ===============================
# sleep 1

# export CUDA_VISIBLE_DEVICES=${gpus[1]}
# echo GPU:$CUDA_VISIBLE_DEVICES

# src_lang=eng
# tgt_lang=spa

# timestamp=$(date +"%Y%m%d-%H.%M.%S")
# echo ${src_lang}' > '${tgt_lang}

# python /fs/surtr0/jprats/code/llm-mt-iberian-languages/src/eval_llm_mt.py \
#     --filename_prefix $filename_prefix \
#     --timestamp $timestamp \
#     --batch_size 8 \
#     --num_beams 5 \
#     --max_new_tokens 150 \
#     --num_fewshot $num_fewshot \
#     --template_id simple \
#     --src_examples /fs/surtr0/jprats/data/raw/flores200_dataset/${example_set}/${src_lang}_Latn.${example_set} \
#     --ref_examples /fs/surtr0/jprats/data/raw/flores200_dataset/${example_set}/${tgt_lang}_Latn.${example_set} \
#     /fs/surtr0/jprats/data/raw/flores200_dataset/${eval_set}/${src_lang}_Latn.${eval_set} \
#     /fs/surtr0/jprats/data/raw/flores200_dataset/${eval_set}/${tgt_lang}_Latn.${eval_set} \
#     /fs/surtr0/jprats/code/llm-mt-iberian-languages \
#     $model 2> /fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/mt_eval/${filename_prefix}_${src_lang}-${tgt_lang}_${num_fewshot}_${timestamp}.log # &

# # ===============================
# sleep 1

# export CUDA_VISIBLE_DEVICES=${gpus[1]}
# echo GPU:$CUDA_VISIBLE_DEVICES

# src_lang=spa
# tgt_lang=cat

# timestamp=$(date +"%Y%m%d-%H.%M.%S")
# echo ${src_lang}' > '${tgt_lang}

# python /fs/surtr0/jprats/code/llm-mt-iberian-languages/src/eval_llm_mt.py \
#     --filename_prefix $filename_prefix \
#     --timestamp $timestamp \
#     --batch_size 8 \
#     --num_beams 5 \
#     --max_new_tokens 150 \
#     --num_fewshot $num_fewshot \
#     --template_id simple \
#     --src_examples /fs/surtr0/jprats/data/raw/flores200_dataset/${example_set}/${src_lang}_Latn.${example_set} \
#     --ref_examples /fs/surtr0/jprats/data/raw/flores200_dataset/${example_set}/${tgt_lang}_Latn.${example_set} \
#     /fs/surtr0/jprats/data/raw/flores200_dataset/${eval_set}/${src_lang}_Latn.${eval_set} \
#     /fs/surtr0/jprats/data/raw/flores200_dataset/${eval_set}/${tgt_lang}_Latn.${eval_set} \
#     /fs/surtr0/jprats/code/llm-mt-iberian-languages \
#     $model 2> /fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/mt_eval/${filename_prefix}_${src_lang}-${tgt_lang}_${num_fewshot}_${timestamp}.log # &

# # ===============================
# sleep 1

# export CUDA_VISIBLE_DEVICES=${gpus[1]}
# echo GPU:$CUDA_VISIBLE_DEVICES

# src_lang=spa
# tgt_lang=eng

# timestamp=$(date +"%Y%m%d-%H.%M.%S")
# echo ${src_lang}' > '${tgt_lang}

# python /fs/surtr0/jprats/code/llm-mt-iberian-languages/src/eval_llm_mt.py \
#     --filename_prefix $filename_prefix \
#     --timestamp $timestamp \
#     --batch_size 8 \
#     --num_beams 5 \
#     --max_new_tokens 150 \
#     --num_fewshot $num_fewshot \
#     --template_id simple \
#     --src_examples /fs/surtr0/jprats/data/raw/flores200_dataset/${example_set}/${src_lang}_Latn.${example_set} \
#     --ref_examples /fs/surtr0/jprats/data/raw/flores200_dataset/${example_set}/${tgt_lang}_Latn.${example_set} \
#     /fs/surtr0/jprats/data/raw/flores200_dataset/${eval_set}/${src_lang}_Latn.${eval_set} \
#     /fs/surtr0/jprats/data/raw/flores200_dataset/${eval_set}/${tgt_lang}_Latn.${eval_set} \
#     /fs/surtr0/jprats/code/llm-mt-iberian-languages \
#     $model 2> /fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/mt_eval/${filename_prefix}_${src_lang}-${tgt_lang}_${num_fewshot}_${timestamp}.log # &

# ===============================
num_fewshot=1
# ===============================

# export CUDA_VISIBLE_DEVICES=${gpus[1]}
# echo GPU:$CUDA_VISIBLE_DEVICES

# src_lang=cat
# tgt_lang=eng

# timestamp=$(date +"%Y%m%d-%H.%M.%S")
# echo ${src_lang}' > '${tgt_lang}

# python /fs/surtr0/jprats/code/llm-mt-iberian-languages/src/eval_llm_mt.py \
#     --filename_prefix $filename_prefix \
#     --timestamp $timestamp \
#     --batch_size 8 \
#     --num_beams 5 \
#     --max_new_tokens 150 \
#     --num_fewshot $num_fewshot \
#     --template_id simple \
#     --src_examples /fs/surtr0/jprats/data/raw/flores200_dataset/${example_set}/${src_lang}_Latn.${example_set} \
#     --ref_examples /fs/surtr0/jprats/data/raw/flores200_dataset/${example_set}/${tgt_lang}_Latn.${example_set} \
#     /fs/surtr0/jprats/data/raw/flores200_dataset/${eval_set}/${src_lang}_Latn.${eval_set} \
#     /fs/surtr0/jprats/data/raw/flores200_dataset/${eval_set}/${tgt_lang}_Latn.${eval_set} \
#     /fs/surtr0/jprats/code/llm-mt-iberian-languages \
#     $model 2> /fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/mt_eval/${filename_prefix}_${src_lang}-${tgt_lang}_${num_fewshot}_${timestamp}.log # &

# # ===============================
# sleep 1

# export CUDA_VISIBLE_DEVICES=${gpus[1]}
# echo GPU:$CUDA_VISIBLE_DEVICES

# src_lang=cat
# tgt_lang=spa

# timestamp=$(date +"%Y%m%d-%H.%M.%S")
# echo ${src_lang}' > '${tgt_lang}

# python /fs/surtr0/jprats/code/llm-mt-iberian-languages/src/eval_llm_mt.py \
#     --filename_prefix $filename_prefix \
#     --timestamp $timestamp \
#     --batch_size 8 \
#     --num_beams 5 \
#     --max_new_tokens 150 \
#     --num_fewshot $num_fewshot \
#     --template_id simple \
#     --src_examples /fs/surtr0/jprats/data/raw/flores200_dataset/${example_set}/${src_lang}_Latn.${example_set} \
#     --ref_examples /fs/surtr0/jprats/data/raw/flores200_dataset/${example_set}/${tgt_lang}_Latn.${example_set} \
#     /fs/surtr0/jprats/data/raw/flores200_dataset/${eval_set}/${src_lang}_Latn.${eval_set} \
#     /fs/surtr0/jprats/data/raw/flores200_dataset/${eval_set}/${tgt_lang}_Latn.${eval_set} \
#     /fs/surtr0/jprats/code/llm-mt-iberian-languages \
#     $model 2> /fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/mt_eval/${filename_prefix}_${src_lang}-${tgt_lang}_${num_fewshot}_${timestamp}.log # &

# ===============================
sleep 1

export CUDA_VISIBLE_DEVICES=${gpus[1]}
echo GPU:$CUDA_VISIBLE_DEVICES

src_lang=eng
tgt_lang=cat

timestamp=$(date +"%Y%m%d-%H.%M.%S")
echo ${src_lang}' > '${tgt_lang}

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

# ===============================
sleep 1

export CUDA_VISIBLE_DEVICES=${gpus[1]}
echo GPU:$CUDA_VISIBLE_DEVICES

src_lang=eng
tgt_lang=spa

timestamp=$(date +"%Y%m%d-%H.%M.%S")
echo ${src_lang}' > '${tgt_lang}

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

# ===============================
sleep 1

export CUDA_VISIBLE_DEVICES=${gpus[1]}
echo GPU:$CUDA_VISIBLE_DEVICES

src_lang=spa
tgt_lang=cat

timestamp=$(date +"%Y%m%d-%H.%M.%S")
echo ${src_lang}' > '${tgt_lang}

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

# ===============================
sleep 1

export CUDA_VISIBLE_DEVICES=${gpus[1]}
echo GPU:$CUDA_VISIBLE_DEVICES

src_lang=spa
tgt_lang=eng

timestamp=$(date +"%Y%m%d-%H.%M.%S")
echo ${src_lang}' > '${tgt_lang}

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

# ===============================
