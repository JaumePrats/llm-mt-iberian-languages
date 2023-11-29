#!/bin/bash
export CUDA_VISIBLE_DEVICES=6
echo GPU:$CUDA_VISIBLE_DEVICES

model=tiiuae/falcon-7b
adapter=/fs/surtr0/jprats/models/falcon_peft_test_1.0-slowTokenizer/checkpoint-1000
eval_set=dev
example_set=devtest
filename_prefix=TEST_directory_structure-$eval_set

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
    --src_examples /fs/surtr0/jprats/data/raw/flores200_dataset/${example_set}/${src_lang}_Latn.${example_set} \
    --ref_examples /fs/surtr0/jprats/data/raw/flores200_dataset/${example_set}/${tgt_lang}_Latn.${example_set} \
    /fs/surtr0/jprats/data/raw/flores200_dataset/${eval_set}/${src_lang}_Latn.${eval_set} \
    /fs/surtr0/jprats/data/raw/flores200_dataset/${eval_set}/${tgt_lang}_Latn.${eval_set} \
    /fs/surtr0/jprats/code/llm-mt-iberian-languages \
    $model \
    2> /fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/${filename_prefix}_${src_lang}-${tgt_lang}_${timestamp}.log