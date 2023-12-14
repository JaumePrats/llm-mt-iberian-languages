#!/bin/bash

model=tiiuae/falcon-7b
prefix=FM_falcon_idioms
nums_fewshot=(0 1)
directions=("eng-spa" "spa-eng")
sets=("idioms" "distractors" "all")
gpus=(7)

flores_example_set=dev

for num_fewshot in "${nums_fewshot[@]}"; do

    for direction in "${directions[@]}"; do

        for set in "${sets[@]}"; do

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
            filename_prefix="${prefix}-${set}"

            echo "---------------------------------------------------"
            echo "IDIOMS - ${set} set: ${filename_prefix}_${src_lang}-${tgt_lang}_${num_fewshot}_${timestamp}.log"
            echo "---------------------------------------------------"
            echo "model: $model"
            echo "Direction: ${src_lang} > ${tgt_lang}"
            echo "idioms set: ${set}"
            echo "num fewhot: $num_fewshot"
            echo "cuda devices: $CUDA_VISIBLE_DEVICES"
            echo "---------------------------------------------------"

            python /fs/surtr0/jprats/code/llm-mt-iberian-languages/src/eval_llm_mt.py \
                --filename_prefix $filename_prefix \
                --timestamp $timestamp \
                --batch_size 8 \
                --num_beams 5 \
                --max_new_tokens 110 \
                --num_fewshot $num_fewshot \
                --template_id simple \
                --src_examples /fs/surtr0/jprats/data/raw/flores200_dataset/${flores_example_set}/${src_lang}_Latn.${flores_example_set} \
                --ref_examples /fs/surtr0/jprats/data/raw/flores200_dataset/${flores_example_set}/${tgt_lang}_Latn.${flores_example_set} \
                /fs/surtr0/jprats/data/raw/idioms_francesca/extraction/idioms_francesca.${set}.${src_lang} \
                /fs/surtr0/jprats/data/raw/idioms_francesca/extraction/idioms_francesca.${set}.${tgt_lang} \
                /fs/surtr0/jprats/code/llm-mt-iberian-languages \
                $model \
                > "/fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/idiom_log.log" \
                2> "/fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/mt_eval/${filename_prefix}_${src_lang}-${tgt_lang}_${num_fewshot}_${timestamp}.log"

        done
    done
done


