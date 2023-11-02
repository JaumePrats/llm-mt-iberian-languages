import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from datetime import datetime

model_id = 'projecte-aina/aguila-7b'
batch_size = 8

src_path = '/fs/surtr0/jprats/data/processed/tiny_flores/eng_Latn.dev'
tgt_path = '/fs/alvis0/jprats/code/llm-mt-iberian-languages/tgt_out/TEST_aguila_example.txt'
complete_out_path = '/fs/alvis0/jprats/code/llm-mt-iberian-languages/complete_out/TEST_aguila_example.txt'

prompt = ['English: <s>', '</s>\nSpanish: <s>']
num_fewshot = 0
print('Example prompt:')
print(prompt[0] + 'Example sentence to translate.' + prompt[1])

start_time = datetime.now()

tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
if tokenizer.pad_token is None: # decoder-only models such as Falcon are not trained with pad_token, so the tokenizer does not have one set.
    tokenizer.pad_token = tokenizer.eos_token
generator = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

# reading src data and doing generation
with open(src_path, 'r') as src_file:
    src_lines = src_file.readlines() # read src
    #Constructing input_text from the test set
    input_text = [''.join([prompt[0], src_line.strip(), prompt[1]]) for src_line in src_lines]
    try:
        generations = generator( # generate tgt sentences
            input_text,
            num_beams = 5,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=70,
            batch_size = batch_size
        )
    except Exception as e:
        print('An error occurred while generating')
        raise e
    
with open(tgt_path, 'w') as tgt_file, open(complete_out_path, 'w') as complete_out_file:
    for generation in generations:
        full_output = generation[0]['generated_text']
        complete_out_file.write(full_output + "\n" + 20*'-' + '\n') # save full output of the model
        tgt_file.write(full_output.split("<s>")[(num_fewshot+1)*2].split("</s>")[0] + '\n') # save stripped sentence

end_time = datetime.now()
exec_time = end_time - start_time
print('Execution time: ', exec_time)
