import torch
import sys
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
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

print('using cuda:', torch.cuda.is_available())

# DIRECTORIES
RESULTS_DIR = 'results/mt_eval/'
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

def main(io_params, model_params):
    
    print('device:', model_params['device'])
    print(type(model_params['device']))

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
    src_filename = io_params['src_data'].split('/')[-1]
    ref_filename = io_params['ref_data'].split('/')[-1]
    for lang_code in LANG_CODES:
        if lang_code in src_filename:
            src_lang_code = lang_code
        if lang_code in ref_filename:
            ref_lang_code = lang_code
    assert src_lang_code != None
    assert ref_lang_code != None
    translation_direction = src_lang_code + '-' + ref_lang_code

    # define filename: <prefix>_<translation_direction>_<model_id>_<template_id>_<num_fewshot>_<date-time>.txt
    filename = f"{io_params['filename_prefix']}_{translation_direction}_{model_params['model_id'].replace('/','-')}_bs{model_params['batch_size']}_{io_params['timestamp']}"

    # initialize logger
    #log_path = os.path.join(PATH_PREFIX, LOG_DIR, filename + '.log')
    sys.stderr
    #file_handler = logging.FileHandler(filename=log_path)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stderr_handler = logging.StreamHandler(stream=sys.stderr)
    #handlers = [file_handler, stdout_handler, stderr_handler]
    handlers = [stdout_handler, stderr_handler]
    logging.basicConfig(level=logging.DEBUG, handlers=handlers)

    # get src tgt BCP-47 codes
    src_bcp47 = src_lang_code + '_Latn'
    tgt_bcp47 = ref_lang_code + '_Latn'

    logging.info('Loading model...')
    # initialize generator
    model = AutoModelForSeq2SeqLM.from_pretrained(model_params['model_id'])
    tokenizer = AutoTokenizer.from_pretrained(model_params['model_id'])
    generator = pipeline(
        "translation",
        model=model,
        tokenizer=tokenizer,
        src_lang=src_bcp47,
        tgt_lang=tgt_bcp47,
        max_length=model_params['max_length'],
        device=model_params['device'],
        batch_size=model_params['batch_size']
    )

    results_path = os.path.join(PATH_PREFIX, RESULTS_DIR, filename + '.txt')
    tgt_path = os.path.join(PATH_PREFIX, TGT_DIR, filename + f'.{ref_lang_code}')

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

    # reading src_data and doing generation
    with open(io_params['src_data'], 'r') as src_file:
        src_lines = src_file.readlines() # read src
        logging.info('Constructing input_text from the test set')
        input_text = [line.strip() for line in src_lines]
        logging.info('Generating...')
        try:
            generations = generator(input_text)
        except Exception as e:
            logging.error('An error occurred while generating')
            logging.error(str(e))
            raise e
    
    logging.info('Saving outputs...')
    with open(tgt_path, 'w') as tgt_file:
        for generation in generations:
            tgt_line = generation['translation_text']
            tgt_file.write(tgt_line + '\n')
    
    # evaluation
    with open(results_path, 'a') as results_file: # opening file with access mode 'a' (append)
        logging.info('Evaluating...')
        results_file.write('\n')
        results_file.write(f"EVALUATION RESULTS: {20*'='}\n")

        # OFF-TARGET TRANSLATION
        results_file.write(f"\nOFF-TARGET TRANSLATION: {10*'-'}\n")
        lg = Lang(ref_lang_code)
        iso_ref_lang = lg.pt1
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
        comet22_score_path = os.path.join(PATH_PREFIX, COMET22_DIR, filename)
        c22_score = comet_score(io_params['src_data'], tgt_path, io_params['ref_data'], model="Unbabel/wmt22-comet-da", score_path=comet22_score_path)
        results_file.write('COMET22 = '+ str(c22_score) + '\n') 
        #comet20
        comet20_score_path = os.path.join(PATH_PREFIX, COMET20_DIR, filename)
        c20_score = comet_score(io_params['src_data'], tgt_path, io_params['ref_data'], model="Unbabel/wmt20-comet-da", score_path=comet20_score_path)
        results_file.write('COMET20 = '+ str(c20_score) + '\n')

        #add line to copy in spreadsheet
        results_file.write(f"\ncopy results (blue comet22 comet20 off-tgt) {10*'-'}\n")
        results_file.write(f"{b_score.split(' ')[2]} {c22_score} {c20_score} {ot_score}\n")


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
    parser.add_argument('--max_length', type=int, default=400, help='Maximum length of the translated outputs')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of test sentences that will run on parallel.')
    parser.add_argument('--device', type=int, help='GPU device on which the model will run.')

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
        'max_length': params.max_length,
        'batch_size': params.batch_size,
        'device': params.device
    }

    main(io_params, model_params)