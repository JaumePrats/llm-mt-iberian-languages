#!/bin/bash

# model=projecte-aina/aguila-7b
model=tiiuae/falcon-7b
eval_set=devtest
example_set=dev
filename_prefix=FM-EVAL_Falcon_flores-$eval_set
nums_fewshot=(5)
# nums_fewshot=(0 1)
# directions=("cat-eng" "cat-spa" "eng-cat" "eng-spa" "spa-cat" "spa-eng")
directions=("cat-eng")
gpus=(0 4)

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

        echo "---------------------------------------------------"
        echo FLORES: ${filename_prefix}_${src_lang}-${tgt_lang}_${num_fewshot}_${timestamp}.log
        echo "---------------------------------------------------"
        echo "model: $model"
        echo "Direction: ${src_lang} > ${tgt_lang}"
        echo "eval set: $eval_set"
        echo "num fewhot: $num_fewshot"
        echo "cuda devices: $CUDA_VISIBLE_DEVICES"
        echo "---------------------------------------------------"

        python /fs/surtr0/jprats/code/llm-mt-iberian-languages/src/eval_llm_mt.py \
            --filename_prefix $filename_prefix \
            --timestamp $timestamp \
            --batch_size 8 \
            --max_new_tokens 150 \
            --num_fewshot $num_fewshot \
            --template_id simple \
            --src_examples /fs/surtr0/jprats/data/raw/flores200_dataset/${example_set}/${src_lang}_Latn.${example_set} \
            --ref_examples /fs/surtr0/jprats/data/raw/flores200_dataset/${example_set}/${tgt_lang}_Latn.${example_set} \
            /fs/surtr0/jprats/data/raw/flores200_dataset/${eval_set}/${src_lang}_Latn.${eval_set} \
            /fs/surtr0/jprats/data/raw/flores200_dataset/${eval_set}/${tgt_lang}_Latn.${eval_set} \
            /fs/surtr0/jprats/code/llm-mt-iberian-languages \
            $model \
            > "/fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/${filename_prefix}_${src_lang}-${tgt_lang}_${num_fewshot}_${timestamp}.log" \
            2> "/fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/mt_eval/${filename_prefix}_${src_lang}-${tgt_lang}_${num_fewshot}_${timestamp}.log"

    done

done


