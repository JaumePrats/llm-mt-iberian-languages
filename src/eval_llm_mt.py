import torch
import sys
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
import json
import argparse
import random
import os
import sys
from tqdm import tqdm
from iso639 import Lang
import logging
from metrics import bleu_score, comet_score, off_target_score
from max_tokens import get_max_token_length
import gc

# DIRECTORIES
RESULTS_DIR = 'results/'
TGT_DIR = 'tgt_out/'
COMPLETE_OUT_DIR = 'complete_out/'
COMET_SCORES_DIR = RESULTS_DIR + 'comet_scores/'
COMET22_DIR = COMET_SCORES_DIR + 'comet22/'
COMET20_DIR = COMET_SCORES_DIR + 'comet20/'
LOG_DIR = 'logs/'

# FILES
LANG_FILE = 'languages.json'
TEMPL_FILE = 'templates.json'

PATH_PREFIX = ''
LANG_CODES = {}

# initialize logger
#log_path = os.path.join(PATH_PREFIX, LOG_DIR, filename + '.log')
sys.stderr
#file_handler = logging.FileHandler(filename=log_path)
stdout_handler = logging.StreamHandler(stream=sys.stdout)
stderr_handler = logging.StreamHandler(stream=sys.stderr)
#handlers = [file_handler, stdout_handler, stderr_handler]
handlers = [stdout_handler, stderr_handler]
logging.basicConfig(level=logging.DEBUG, handlers=handlers)

def extract_examples(src_path, ref_path, num_fewshot, selected = False):
    '''
    Extract 'num_fewshot' examples from the source and target file.

    :return: list
        A list with each of the examples as [[src_ex1, ref_ex1], [src_ex2, ref_ex2], ...]

    '''
    if selected == True:
        print('Using selected examples for in-context learning')
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

    templates_path = os.path.join(PATH_PREFIX, TEMPL_FILE)
    with open(templates_path, 'r') as f:
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

def translate(io_params: dict, model_params: dict, prompt_params: dict, prompt: list, tgt_path: str, complete_out_path: str):
    import pdb; pdb.set_trace()

    logging.info('Loading model...')
    # initialize generator
    tokenizer = AutoTokenizer.from_pretrained(model_params['model_id'], padding_side='left')
    max_ref_len, avg_ref_len, min_ref_len = get_max_token_length(io_params['ref_data'], tokenizer) # check max tokens in ref file
    if max_ref_len + 3 >= model_params['max_new_tokens']: # 3 tokens for </s>
        logging.error("The output of the model may be cut short. Consider increasing 'max_new_tokens' value.")
    logging.info(f"max_new_tokens={model_params['max_new_tokens']}")
    logging.info(f'max tokens on ref sentence: {max_ref_len}')

    if tokenizer.pad_token is None: # decoder-only models such as Falcon are not trained with pad_token, so the tokenizer does not have one set.
        tokenizer.pad_token = tokenizer.eos_token
    generator = pipeline(
        "text-generation",
        model=model_params['model_id'],
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    # reading src_data and doing generation
    with open(io_params['src_data'], 'r') as src_file:
        src_lines = src_file.readlines() # read src
        logging.info('Constructing input_text from the test set')
        input_text = [''.join([prompt[0], src_line.strip(), prompt[1]]) for src_line in src_lines]
        logging.info('Generating...')
        try:
            generations = generator( # generate tgt sentences
                input_text,
                num_beams = model_params['num_beams'],
                do_sample=model_params['do_sample'],
                top_k=model_params['top_k'],
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=model_params['max_new_tokens'],
                batch_size = model_params['batch_size']
            )
        except Exception as e:
            logging.error('An error occurred while generating')
            logging.error(str(e))
            raise e
    
    logging.info('Saving outputs...')
    with open(tgt_path, 'w') as tgt_file, open(complete_out_path, 'w') as complete_out_file:
        for generation in generations:
            full_output = generation[0]['generated_text']
            complete_out_file.write(full_output + "\n" + 20*'-' + '\n') # save full output of the model
            tgt_file.write(full_output.split("<s>")[(prompt_params['num_fewshot']+1)*2].split("</s>")[0] + '\n') # save stripped sentence

def evaluate(tgt_path: str, ref_lang_code: str, results_path):

    comet_filename = os.path.splitext(os.path.basename(results_path))[0]

    with open(results_path, 'a') as results_file: # opening file with access mode 'a' (append)
        logging.info('Evaluating...')
        results_file.write('\n')
        results_file.write(f"EVALUATION RESULTS: {20*'='}\n")

        # OFF-TARGET TRANSLATION
        results_file.write(f"\nOFF-TARGET TRANSLATION: {10*'-'}\n")
        ot_score, ot_stats = off_target_score(tgt_path, ref_lang_code, return_tgt_langs_stats=True)
        results_file.write('OFF-TGT(%) = '+ str(ot_score) + '\n')
        results_file.write(json.dumps(ot_stats, sort_keys=True, indent=4) + '\n')

        # BLEU
        results_file.write(f"\nBLEU: {10*'-'}\n")
        b_score, b_signature = bleu_score(tgt_path, io_params['ref_data'])
        results_file.write(b_score + '\n')
        results_file.write('Signature: '+ str(b_signature) + '\n')

        # COMET
        results_file.write(f"\nCOMET: {10*'-'}\n")
        # comet22
        comet22_score_path = os.path.join(PATH_PREFIX, COMET22_DIR, comet_filename)
        c22_score = comet_score(io_params['src_data'], tgt_path, io_params['ref_data'], model="Unbabel/wmt22-comet-da", score_path=comet22_score_path)
        results_file.write('COMET22 = '+ str(c22_score) + '\n') 
        #comet20
        comet20_score_path = os.path.join(PATH_PREFIX, COMET20_DIR, comet_filename)
        c20_score = comet_score(io_params['src_data'], tgt_path, io_params['ref_data'], model="Unbabel/wmt20-comet-da", score_path=comet20_score_path)
        results_file.write('COMET20 = '+ str(c20_score) + '\n')

        #add line to copy in spreadsheet
        results_file.write(f"\ncopy results (blue comet22 comet20 off-tgt) {10*'-'}\n")
        results_file.write(f"{b_score.split(' ')[2]} {c22_score} {c20_score} {ot_score}\n")

def main(io_params, model_params, prompt_params):

    # initialize constants (PATH_PREFIX, LANG_CODES)
    global PATH_PREFIX
    PATH_PREFIX = io_params['path_prefix']

    global LANG_CODES
    lang_path = os.path.join(PATH_PREFIX, LANG_FILE)
    with open(lang_path, 'r') as f:
        LANG_CODES = json.load(f)

    # get time
    start_time = datetime.now()
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
    filename = f"{io_params['filename_prefix']}_{translation_direction}_{model_params['model_id'].replace('/','-')}_{prompt_params['template_id']}_nshot{prompt_params['num_fewshot']}_bs{model_params['batch_size']}_{io_params['timestamp']}"

    results_path = os.path.join(PATH_PREFIX, RESULTS_DIR, filename + '.txt')
    tgt_path = os.path.join(PATH_PREFIX, TGT_DIR, filename + f'.{ref_lang_code}')
    complete_out_path = os.path.join(PATH_PREFIX, COMPLETE_OUT_DIR, filename + '.txt')

    logging.info('Creating prompt')
    prompt = create_prompt(prompt_params['num_fewshot'], prompt_params['template_id'], prompt_params['src_examples'], prompt_params['ref_examples'])
    logging.debug(f"PROMT:\n{prompt}")

    translate(
        io_params=io_params,
        model_params=model_params,
        prompt_params=prompt_params,
        prompt=prompt,
        tgt_path=tgt_path,
        complete_out_path=complete_out_path
              )
    
    # remove llm model from GPU to avoid problems with evaluation
    torch.cuda.empty_cache()
    gc.collect()

    # save used parameters
    logging.info('Saving parameters')
    with open(results_path, 'w') as results_file:

        # writing test parameters in results file
        results_file.write(f"TEST PARAMETERS: {10*'-'}\n")
        results_file.write(f"start time: {start_time.strftime('%d/%m/%Y at %H:%M:%S')}\n")
        results_file.write(f"execution time: - (in progress...)\n")
        exec_time_line = 2
        results_file.write(f"translation direction: {translation_direction}\n")
        results_file.write(f"IO PARAMETERS: {10*'-'}\n")
        results_file.write(json.dumps(io_params, sort_keys=True, indent=4) + '\n')
        results_file.write(f"MODEL PARAMETERS: {10*'-'}\n")
        results_file.write(json.dumps(model_params, sort_keys=True, indent=4) + '\n')
        results_file.write(f"PROMPT PARAMETERS: {10*'-'}\n")
        results_file.write(json.dumps(prompt_params, sort_keys=True, indent=4) + '\n')
    
    # evaluation
    evaluate(
        tgt_path=tgt_path,
        ref_lang_code=ref_lang_code,
        results_path=results_path
    )

    # computing and saving execution time
    end_time = datetime.now()
    exec_time = end_time - start_time
    with open(results_path, 'r') as results_file:
        lines = results_file.readlines()
    lines[exec_time_line] = f"execution time: {str(exec_time)}\n"
    with open(results_path, 'w') as results_file: 
        results_file.writelines(lines)
    logging.info(f'Evaluation finished. Total execution time: {str(exec_time)}')
    

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Evaluate LLM on test set.')

    # io parameters
    parser.add_argument('src_data', type=str, help='path to src file of the test set')
    parser.add_argument('ref_data', type=str, help='path to ref file of the test set (used for evaluation)')
    parser.add_argument('path_prefix', type=str, default='', help='prefix of the repository directory. example: /home/usr/code/llm-mt-iberian-languages')
    parser.add_argument('--filename_prefix', type=str, default='', help='prefix for the log filename. Log filename: <prefix>_<model_id>_<template_id>_<num_fewshot>_<date-time>.txt')
    parser.add_argument('--timestamp', type=str, help='Time of execution, used to relate error and output files')

    # model parameters
    parser.add_argument('model_id', type=str, help='HF id of the model to run, example:tiiuae/falcon-7b')
    parser.add_argument('--num_beams', type=int, default=1, help='Number of beams for beam search. 1 means no beam search')
    parser.add_argument('--do_sample', type=bool, default=False, help='Whether or not to use sampling ; use greedy decoding otherwise')
    parser.add_argument('--top_k', type=int, default=50, help='The number of highest probability vocabulary tokens to keep for top-k-filtering')
    parser.add_argument('--max_new_tokens', type=int, default=50, help='The minimum numbers of tokens to generate, ignoring the number of tokens in the prompt.')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of test sentences that will run on parallel.')

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
        'filename_prefix': params.filename_prefix,
        'timestamp': params.timestamp
    }

    model_params = {
        'model_id': params.model_id,
        'num_beams': params.num_beams,
        'do_sample': params.do_sample,
        'top_k': params.top_k,
        'max_new_tokens': params.max_new_tokens,
        'batch_size': params.batch_size
    }

    prompt_params = {
        'template_id': params.template_id,
        'num_fewshot': params.num_fewshot,
        'src_examples': params.src_examples,
        'ref_examples': params.ref_examples,
    }

    main(io_params, model_params, prompt_params)