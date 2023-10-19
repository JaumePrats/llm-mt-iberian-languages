import torch
import sys
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import time
import json
import argparse
import random

def extract_examples(src_path, ref_path, num_fewshot, random = True):
    '''
    Extract 'num_fewshot' examples from the source and target file.

    :return: list
        A list with each of the examples as [[src_ex1, ref_ex1], [src_ex2, ref_ex2], ...]

    '''
    if random != True:
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

    with open('languages.json', 'r') as (f, err):
        if err:
            print(err)
        else:
            languages_codes = json.load(f)

    with open('templates.json', 'r') as (f, err):
        if err:
            print(err)
        else:
            templates = json.load(f)
            template = templates[template_id]
    
    # get src and tgt languages from files
    src_filename = src_examples.split('/')[-1]
    ref_filename = ref_examples.split('/')[-1]
    for lang_code in languages_codes:
        if src_filename.contains(lang_code):
            src_language = languages_codes[lang_code]
        if ref_filename.contains(lang_code):
            ref_language = languages_codes[lang_code]
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


def main(general_params, prompt_params):
    prompt = create_prompt(prompt_params.num_fewshot, prompt_params.template_id, prompt_params.src_examples, prompt_params.ref_examples)

    #TODO
    

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Run LLM on test set')

    parser.add_argument('model', type=str, help='HF id of the model to run, example:tiiuae/falcon-7b')
    parser.add_argument('src_data', type=str, help='path to src file of the test set')
    parser.add_argument('tgt_data', type=str, help='path where tgt data will be stored')
    parser.add_argument('--num_fewshot', type=int, default=0, help='number of examples given to the model')
    parser.add_argument('--template_id', type=str, default='simple', help='name of the prompt used, see file templates.json')
    parser.add_argument('--src_examples', type=str, default=None, help='source of the examples used in prompt')
    parser.add_argument('--ref_examples', type=str, default=None, help='reference of the examples used in prompt')
    parser.add_argument('--complete_output', type=str, default=None, help='path where the full output of the model will be stored')

    params=parser.parse_args()

    general_params = {
        'model': params.model,
        'src_data': params.src_data,
        'tgt_data': params.tgt_data,
        'complete_output': params.complete_output
    }

    prompt_params = {
        'template_id': params.template,
        'num_fewshot': params.num_fewshot,
        'src_examples': params.src_examples,
        'ref_examples': params.ref_examples,
    }

    main(general_params, prompt_params)
