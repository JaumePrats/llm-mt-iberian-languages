#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
echo $CUDA_VISIBLE_DEVICES

prefix=FM-EVAL_Falcon_unpc

python /fs/surtr0/jprats/code/llm-mt-iberian-languages/src/reprocess_outputs.py \
    --file-prefix $prefix \
    --results-dir /fs/surtr0/jprats/code/llm-mt-iberian-languages/results/mt_eval/foundation_models \
    --data-dir /fs/surtr0/jprats/data/processed/evaluation/UNPC/testset \
    > "/fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/mt_eval/REPROCESS_${prefix}.log" \
    2> "/fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/mt_eval/REPROCESS_${prefix}.log"