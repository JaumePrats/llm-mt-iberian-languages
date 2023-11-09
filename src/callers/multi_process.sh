#!/bin/bash
export CUDA_VISIBLE_DEVICES=6
echo $CUDA_VISIBLE_DEVICES

python /fs/surtr0/jprats/code/llm-mt-iberian-languages/playground.py &

export CUDA_VISIBLE_DEVICES=7
echo $CUDA_VISIBLE_DEVICES

python /fs/surtr0/jprats/code/llm-mt-iberian-languages/playground.py &