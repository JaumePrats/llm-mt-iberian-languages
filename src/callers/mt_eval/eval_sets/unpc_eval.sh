#!/bin/bash

# model=tiiuae/falcon-7b
model=projecte-aina/aguila-7b
# prefix=PD-QLORA_Falcon_unpc
prefix=FM-EVAL_Aguila_unpc
nums_fewshot=(0 1)
# nums_fewshot=(0)
directions=("eng-spa" "spa-eng")
gpus=(7)

unpc_eval_set=testset
unpc_example_set=devset

for num_fewshot in "${nums_fewshot[@]}"; do

    for direction in "${directions[@]}"; do
        # echo $direction
        IFS='-'
        read -ra direction_array <<< "$direction"
        IFS=$' \t\n'
        src_lang=${direction_array[0]}
        tgt_lang=${direction_array[1]}

        if [ "$num_fewshot" -eq 5 ]; then
            export CUDA_VISIBLE_DEVICES=${gpus[0]},${gpus[1]}
        else
            export CUDA_VISIBLE_DEVICES=${gpus[0]}
        fi

        timestamp=$(date +"%Y%m%d-%H.%M.%S")
        filename_prefix="${prefix}_${unpc_eval_set}"

        echo "---------------------------------------------------"
        echo UNPC: ${filename_prefix}_${src_lang}-${tgt_lang}_${num_fewshot}_${timestamp}.log
        echo "---------------------------------------------------"
        echo "model: $model"
        echo "direction: ${src_lang} > ${tgt_lang}"
        echo "num fewhot: $num_fewshot"
        echo "cuda devices: $CUDA_VISIBLE_DEVICES"
        echo "---------------------------------------------------"

        python /fs/surtr0/jprats/code/llm-mt-iberian-languages/src/eval_llm_mt.py \
            --filename_prefix $filename_prefix \
            --timestamp $timestamp \
            --batch_size 8 \
            --max_new_tokens 300 \
            --num_fewshot $num_fewshot \
            --template_id simple \
            --src_examples /fs/surtr0/jprats/data/processed/evaluation/UNPC/${unpc_example_set}/UNv1.0.${unpc_example_set}.${src_lang} \
            --ref_examples /fs/surtr0/jprats/data/processed/evaluation/UNPC/${unpc_example_set}/UNv1.0.${unpc_example_set}.${tgt_lang} \
            /fs/surtr0/jprats/data/processed/evaluation/UNPC/${unpc_eval_set}/UNv1.0.${unpc_eval_set}.${src_lang} \
            /fs/surtr0/jprats/data/processed/evaluation/UNPC/${unpc_eval_set}/UNv1.0.${unpc_eval_set}.${tgt_lang} \
            /fs/surtr0/jprats/code/llm-mt-iberian-languages \
            $model \
            > /fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/${filename_prefix}_${src_lang}-${tgt_lang}_${num_fewshot}_${timestamp}.log \
            2> /fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/mt_eval/${filename_prefix}_${src_lang}-${tgt_lang}_${num_fewshot}_${timestamp}.log # &

    done

done