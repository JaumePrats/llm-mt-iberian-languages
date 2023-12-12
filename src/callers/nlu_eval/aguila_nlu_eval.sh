prefix=WNLI
model=projecte-aina/aguila-7b
tasks=wnli_ca
num_fewshot=0
device=6

timestamp=$(date +"%Y%m%d-%H.%M.%S")

echo "---------------------------------------------------"
echo NLU EVAL: ${prefix}_${tasks//,/-}_nshot${num_fewshot}_${timestamp}
echo "---------------------------------------------------"
echo "model: $model"
echo "tasks: $tasks"
echo "num fewhot: $num_fewshot"
echo "cuda device: $device"
echo "---------------------------------------------------"

python /fs/surtr0/jprats/code/lm-evaluation-harness/main.py \
    --model hf-causal \
    --model_args pretrained=$model,dtype=float16,trust_remote_code=True \
    --tasks $tasks \
    --num_fewshot $num_fewshot \
    --device cuda:${device} \
    > /fs/surtr0/jprats/code/llm-mt-iberian-languages/results/nlu_eval/${prefix}_"${tasks//,/-}"_nshot${num_fewshot}_${timestamp} \
    # 2> /fs/surtr0/jprats/code/llm-mt-iberian-languages/logs/nlu_eval/${prefix}_"${tasks//,/-}"_nshot${num_fewshot}_${timestamp}