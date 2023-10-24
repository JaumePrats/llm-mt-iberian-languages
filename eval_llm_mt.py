import torch
import sys
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import time
import json
import argparse
import random
import os
from tqdm import tqdm
from metrics import bleu_score, comet_score

# DIRECTORIES
RESULTS_DIR = 'results/'
TGT_DIR = 'tgt_out/'
COMPLETE_OUT_DIR = 'complete_out/'
COMET_SCORES_DIR = RESULTS_DIR + 'comet_scores/'

with open('languages.json', 'r') as f:
    LANG_CODES = json.load(f)

def extract_examples(src_path, ref_path, num_fewshot, selected = False):
    '''
    Extract 'num_fewshot' examples from the source and target file.

    :return: list
        A list with each of the examples as [[src_ex1, ref_ex1], [src_ex2, ref_ex2], ...]

    '''
    if selected == True:
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

    else:
        examples = []
        with open(src_path, 'r') as src_file, open(ref_path, 'r') as ref_file:
            src_lines = src_file.readlines()
            ref_lines = ref_file.readlines()
            if len(ref_lines) != len(src_lines):
                raise ValueError("The number of lines from the examples' source and target files DO NOT MATCH!")
            random_line_indices = random.sample(range(len(src_lines)), num_fewshot) # sample 'num_fewshot' different random lines
            for i in random_line_indices:
                src_line = src_lines[i].strip()
                ref_line = ref_lines[i].strip()
                examples.append([src_line, ref_line])
    return examples

def create_prompt(num_fewshot, template_id, src_examples, ref_examples):

    with open('templates.json', 'r') as f:
        templates = json.load(f)
        template = templates[template_id]
    
    # get src and tgt languages from files
    src_filename = src_examples.split('/')[-1]
    ref_filename = ref_examples.split('/')[-1]
    for lang_code in LANG_CODES:
        if lang_code in src_filename:
            src_language = LANG_CODES[lang_code]
        if lang_code in ref_filename:
            ref_language = LANG_CODES[lang_code]
    assert src_language != None
    assert ref_language != None

    # change <src> and <tgt> for the source and target langauges
    for key, value in template.items():
        template[key] = value.replace('<src>', src_language).replace('<tgt>', ref_language)

    # extract examples
    examples = extract_examples(src_path=src_examples, ref_path=ref_examples, num_fewshot=num_fewshot)

    # create prompt
    prompt = ["",""]
    for i in range(num_fewshot):
        prompt[0] += template['instruction'] + template['question_word'] + ': ' + template['bos'] + examples[i][0] + template['eos'] + '\n' + template['answer_word'] + ': ' + template['bos'] + examples[i][1] + template['eos'] + '\n\n'
    prompt[0] += template['instruction'] + template['question_word'] + ': ' + template['bos']
    prompt[1] = template['eos'] + '\n' + template['answer_word'] + ': ' + template['bos']

    return prompt


def main(io_params, model_params, prompt_params):
    prompt = create_prompt(prompt_params['num_fewshot'], prompt_params['template_id'], prompt_params['src_examples'], prompt_params['ref_examples'])
    print(f"PROMT:\n{prompt}")

    # initialize generator
    tokenizer = AutoTokenizer.from_pretrained(model_params['model_id'])
    generator = pipeline(
        "text-generation",
        model=model_params['model_id'],
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    # get time
    t = time.localtime()
    filename_time = time.strftime("%Y%m%d-%H.%M.%S", t)
    # get translation direction
    src_filename = prompt_params['src_examples'].split('/')[-1]
    ref_filename = prompt_params['ref_examples'].split('/')[-1]
    for lang_code in LANG_CODES:
        if lang_code in src_filename:
            src_lang_code = lang_code
        if lang_code in ref_filename:
            ref_lang_code = lang_code
    assert src_lang_code != None
    assert ref_lang_code != None
    translation_direction = src_lang_code + '-' + ref_lang_code

    # define filename: <prefix>_<translation_direction>_<model_id>_<template_id>_<num_fewshot>_<date-time>.txt
    filename = f"{io_params['filename_prefix']}_{translation_direction}_{model_params['model_id'].replace('/','-')}_{prompt_params['template_id']}_{prompt_params['num_fewshot']}_{filename_time}.txt"

    results_path = os.path.join(io_params['path_prefix'], RESULTS_DIR, filename)
    tgt_path = os.path.join(io_params['path_prefix'], TGT_DIR, filename)
    complete_out_path = os.path.join(io_params['path_prefix'], COMPLETE_OUT_DIR, filename)

    # save used parameters
    with open(results_path, 'w') as results_file:

        # writing test parameters in results file
        results_file.write(f"TEST PARAMETERS: {10*'-'}\n")
        results_file.write(f"start time: {filename_time}\n")
        results_file.write(f"translation direction: {translation_direction}\n")
        results_file.write(f"IO PARAMETERS: {10*'-'}\n")
        results_file.write(json.dumps(io_params, sort_keys=True, indent=4) + '\n')
        results_file.write(f"MODEL PARAMETERS: {10*'-'}\n")
        results_file.write(json.dumps(model_params, sort_keys=True, indent=4) + '\n')
        results_file.write(f"PROMPT PARAMETERS: {10*'-'}\n")
        results_file.write(json.dumps(prompt_params, sort_keys=True, indent=4) + '\n')

    # reading src_data and doing generation
    with open(io_params['src_data'], 'r') as src_file, open(tgt_path, 'w') as tgt_file, open(complete_out_path, 'w') as complete_out_file:
        src_lines = src_file.readlines() # read src
        for src_line in tqdm(src_lines, unit='line'):
            input_text = ''.join([prompt[0], src_line.strip(), prompt[1]])
            generation = generator( # generate tgt
                input_text,
                num_beams = model_params['num_beams'],
                do_sample=model_params['do_sample'],
                top_k=model_params['top_k'],
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=model_params['max_new_tokens']
            )
            full_output = generation[0]['generated_text']
            complete_out_file.write(full_output + "\n" + 20*'-' + '\n') # save full output of the model
            tgt_file.write(full_output.split("<s>")[(prompt_params['num_fewshot']+1)*2].split("</s>")[0] + '\n') # save stripped sentence
    
    # evaluation
    with open(results_path, 'a') as results_file: # opening file with access mode 'a' (append)
        results_file.write('\n')
        results_file.write(f"EVALUATION RESULTS: {20*'='}\n")
        # BLEU
        results_file.write(f"\nBLEU: {10*'-'}\n")
        b_score, b_signature = bleu_score(tgt_path, io_params['ref_data'])
        results_file.write(b_score + '\n')
        results_file.write('Signature: '+ str(b_signature) + '\n')
        # COMET
        results_file.write(f"\nCOMET: {10*'-'}\n")
        comet_score_path = results_path = os.path.join(io_params['path_prefix'], COMET_SCORES_DIR, filename)
        c_score = comet_score(io_params['src_data'], tgt_path, io_params['ref_data'], comet_score_path)
        results_file.write('COMET = '+ str(c_score) + '\n') 

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Evaluate LLM on test set.')

    # io parameters
    parser.add_argument('src_data', type=str, help='path to src file of the test set')
    parser.add_argument('ref_data', type=str, help='path to ref file of the test set (used for evaluation)')
    parser.add_argument('path_prefix', type=str, default='', help='prefix of the repository directory. example: /home/usr/code/llm-mt-iberian-languages')
    parser.add_argument('--filename_prefix', type=str, default='', help='prefix for the log filename. Log filename: <prefix>_<model_id>_<template_id>_<num_fewshot>_<date-time>.txt')

    # model parameters
    parser.add_argument('model_id', type=str, help='HF id of the model to run, example:tiiuae/falcon-7b')
    parser.add_argument('--num_beams', type=int, default=1, help='Number of beams for beam search. 1 means no beam search')
    parser.add_argument('--do_sample', type=bool, default=False, help='Whether or not to use sampling ; use greedy decoding otherwise')
    parser.add_argument('--top_k', type=int, default=50, help='The number of highest probability vocabulary tokens to keep for top-k-filtering')
    parser.add_argument('--max_new_tokens', type=int, default=50, help='The minimum numbers of tokens to generate, ignoring the number of tokens in the prompt.')

    # prompt parameters
    parser.add_argument('--num_fewshot', type=int, default=0, help='number of examples given to the model')
    parser.add_argument('--template_id', type=str, default='simple', help='name of the prompt used, see file templates.json')
    parser.add_argument('--src_examples', type=str, default=None, help='source of the examples used in prompt')
    parser.add_argument('--ref_examples', type=str, default=None, help='reference of the examples used in prompt')

    params=parser.parse_args()

    io_params = {
        'src_data': params.src_data,
        'ref_data': params.ref_data,
        'path_prefix': params.path_prefix,
        'filename_prefix': params.filename_prefix
    }

    model_params = {
        'model_id': params.model_id,
        'num_beams': params.num_beams,
        'do_sample': params.do_sample,
        'top_k': params.top_k,
        'max_new_tokens': params.max_new_tokens
    }

    prompt_params = {
        'template_id': params.template_id,
        'num_fewshot': params.num_fewshot,
        'src_examples': params.src_examples,
        'ref_examples': params.ref_examples,
    }

    main(io_params, model_params, prompt_params)