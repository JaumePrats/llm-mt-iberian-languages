import torch
import sys
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import time

# PROMPT PARAMS:
bos = "<s>"
eos = "</s>"
source_language = "English"
target_language = "Spanish"
question_word = "English"
answer_word = "Spanish"
num_fewshot = 5 

instruction = 'Translate the following sentence from ' + source_language + ' to ' + target_language + ': \n'

# EXAMPLES:
example1 = ['''The feathers' structure suggests that they were not used in flight but rather for temperature regulation or display. The researchers suggested that, even though this is the tail of a young dinosaur, the sample shows adult plumage and not a chick's down.''',
            '''La estructura que presenta el plumaje sugiere que su función no estaba relacionada con el vuelo, sino que las usaban para regular la temperatura o como indicador de la misma. Los investigadores sostienen que, aunque se trata de la cola de un dinosaurio joven, la muestra analizada presenta rasgos del plumaje de un adulto y no de un polluelo.''']
example2 = ['''They found the Sun operated on the same basic principles as other stars: The activity of all stars in the system was found to be driven by their luminosity, their rotation, and nothing else.''',
            '''Se descubrió que el sol se regía por los mismos principios básicos que otras estrellas: los únicos factores que impulsaban su actividad dentro del sistema eran su luminosidad y su rotación.''']
example3 = ['''The speeds of 802.11n are substantially faster than that of its predecessors with a maximum theoretical throughput of 600Mbit/s.''',
            '''Las velocidades del estándar 802.11n son mucho más altas que las alcanzadas por los que lo precedieron, con un rendimiento teórico máximo de 600 Mbps''']
example4 = ['''Over four million people went to Rome to attend the funeral.''',
            '''Más de cuatro millones de individuos se concentraron en Roma para presenciar el funeral.''']
example5 = ['''Mrs. Kirchner announced her intention to run for president at the Argentine Theatre, the same location she used to start her 2005 campaign for the Senate as member of the Buenos Aires province delegation.''',
            '''El Teatro Argentino fue el lugar donde la señora Kirchner anunció su intención de candidatearse como presidenta; este es el mismo sitio donde inició su campaña para el senado en el año 2005, en representación de la provincia de Buenos Aires.''']
examples = [example1, example2, example3, example4, example5]

# prompt construction:
prompt = ["",""]
for i in range(num_fewshot):
    prompt[0] += instruction + question_word + ': ' + bos + examples[i][0] + eos + '\n' + answer_word + ': ' + bos + examples[i][1] + eos + '\n\n'
prompt[0] += instruction + question_word + ': ' + bos
prompt[1] = eos + '\n' + answer_word + ': ' + bos


model_id  = "projecte-aina/aguila-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
generator = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

# GENERATOR PARAMS:
do_sample = True
top_k = 1
max_new_tokens = 50

t = time.localtime()
filename_time = time.strftime("%Y%m%d-%H.%M.%S", t)

with open("/fs/alvis0/jprats/code/logs/config_test_aguila_mt_" + filename_time + ".txt", "w") as f:
    f.write(f" \
            PROMPT PARAMS: \n \
            num_fewshot = {num_fewshot}\n \
            bos = {bos}\n \
            eos = {eos}\n \
            source_language = {source_language}\n \
            target_language = {target_language}\n \
            question_word = {question_word}\n \
            answer_word = {answer_word}\n \
            instruction = {instruction}\n\n \
            GENERATOR PARAMS: \n \
            do_sample = {do_sample}\n \
            top_k = {top_k}\n \
            max_new_tokens = {max_new_tokens}\n \
            ")

with open("/fs/alvis0/jprats/out_data/test_flores_tr.es.full_output","w") as f:
    for line in sys.stdin:
        
        input_text = ''.join([prompt[0], line.strip(), prompt[1]])
        generation = generator(
        input_text,
        do_sample=do_sample,
        top_k=top_k,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens
        )
        full_output = generation[0]['generated_text']
        f.write(full_output + "\n")
        print(full_output.split("<s>")[(num_fewshot+1)*2].split("</s>")[0])