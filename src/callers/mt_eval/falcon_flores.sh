#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
echo GPU:$CUDA_VISIBLE_DEVICES

model=tiiuae/falcon-7b
adapter=/fs/surtr0/jprats/models/checkpoints/falcon_qlora_europarl10k_NOgbl_ebs16_linear_lr1e-4_20231128-12.59.27/checkpoint-500
eval_set=devtest
example_set=dev
filename_prefix=study-gbl-disabled_$eval_set

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
    --num_fewshot 0 \
    --template_id simple \
    --adapter $adapter \
    --src_examples /fs/surtr0/jprats/data/raw/flores200_dataset/${example_set}/${src_lang}_Latn.${example_set} \
    --ref_examples /fs/surtr0/jprats/data/raw/flores200_dataset/${example_set}/${tgt_lang}_Latn.${example_set} \
    /fs/surtr0/jprats/data/raw/flores200_dataset/${eval_set}/${src_lang}_Latn.${eval_set} \
    /fs/surtr0/jprats/data/raw/flores200_dataset/${eval_set}/${tgt_lang}_Latn.${eval_set} \
    /fs/surtr0/jprats/code/llm-mt-iberian-languages \
    $model \
    2> /fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/mt_eval/${filename_prefix}_${src_lang}-${tgt_lang}_${timestamp}.log