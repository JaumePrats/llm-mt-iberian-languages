python /fs/surtr0/jprats/code/lm-evaluation-harness/main.py \
    --model hf-causal \
    --model_args pretrained=tiiuae/falcon-7b \
    --tasks xnli_ca \
    --num_fewshot 0 \
    --device cuda:6