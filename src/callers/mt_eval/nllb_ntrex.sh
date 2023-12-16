#!/bin/bash

filename_prefix=NLLB_ntrex
directions=("cat-eng" "cat-spa" "eng-cat" "eng-spa" "spa-cat" "spa-eng")
device=4
model=facebook/nllb-200-3.3B
flores_example_set=dev

for direction in "${directions[@]}"; do
    # echo $direction
    IFS='-'
    read -ra direction_array <<< "$direction"
    IFS=$' \t\n'
    src_lang=${direction_array[0]}
    tgt_lang=${direction_array[1]}

    timestamp=$(date +"%Y%m%d-%H.%M.%S")

    echo "---------------------------------------------------"
    echo "NLLB NTREX: ${filename_prefix}_${src_lang}-${tgt_lang}_${num_fewshot}_${timestamp}.log"
    echo "---------------------------------------------------"
    echo "model: $model"
    echo "direction: ${src_lang} > ${tgt_lang}"
    echo "cuda device: $device"
    echo "---------------------------------------------------"

    python /fs/surtr0/jprats/code/llm-mt-iberian-languages/src/eval_nllb_mt.py \
        --filename_prefix $filename_prefix \
        --timestamp $timestamp \
        --batch_size 1 \
        --max_length 400 \
        --device $device \
        /fs/surtr0/jprats/data/processed/evaluation/NTREX/newstest2019-ref.${src_lang}.txt \
        /fs/surtr0/jprats/data/processed/evaluation/NTREX/newstest2019-ref.${tgt_lang}.txt \
        /fs/surtr0/jprats/code/llm-mt-iberian-languages \
        $model \
        >  /fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/${filename_prefix}_${src_lang}-${tgt_lang}_${num_fewshot}_${timestamp}.log \
        2> /fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/mt_eval/${filename_prefix}_${src_lang}-${tgt_lang}_${num_fewshot}_${timestamp}.log

done