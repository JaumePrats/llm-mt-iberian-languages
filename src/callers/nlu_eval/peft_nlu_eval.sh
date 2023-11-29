prefix=TEST_adapter
model=tiiuae/falcon-7b
adapter=/fs/surtr0/jprats/models/finetuned/falcon_peft_test_1.0-slowTokenizer/checkpoint-1000
tasks=xnli_ca,xnli_en,xnli_es
num_fewshot=0
device=5

echo "NLU EVAL --------------------------------"
echo "model: $model (adapter: $adapter)"
echo "tasks: $tasks"
echo "num fewhot: $model"
echo "cuda device: $device"
echo "-----------------------------------------"

timestamp=$(date +"%Y%m%d-%H.%M.%S")

echo 
python /fs/surtr0/jprats/code/lm-evaluation-harness/main.py \
    --model hf-causal \
    --model_args pretrained=$model,peft=$adapter,dtype=torch.bfloat16,trust_remote_code=True \
    --tasks $tasks \
    --num_fewshot 0 \
    --device cuda:${device} \
    > /fs/surtr0/jprats/code/llm-mt-iberian-languages/results/nlu_eval/${prefix}_"${tasks//,/-}"_nshot${num_fewshot}_${timestamp} \
    # 2> /fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/nlu_eval/${prefix}_"${tasks//,/-}"_nshot${num_fewshot}_${timestamp}