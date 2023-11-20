import json
import os
from iso639 import Lang

# IN
flores_base_path = '/fs/surtr0/jprats/data/raw/flores200_dataset'
flores_split = 'dev'
src_lang = 'spa_Latn'
ref_lang = 'eng_Latn'
instruction_template = "###SRC"
response_template = "###TGT"
template = f"{instruction_template} [src_lang]: <s>[src_sentence]</s>\n {response_template} [ref_lang]: <s>[ref_sentence]</s>"

# OUT
output_dir = '/fs/surtr0/jprats/data/processed/template_tests/SRC-TGT'
output_prefix = 'flores'

# -----------------------

# reading input files
src_path = src_path = os.path.join(flores_base_path, flores_split, src_lang + '.' + flores_split)
with open(src_path, 'r') as src_file:
    src_sentences = [line.strip() for line in src_file.readlines()]
    print(f'reading {src_path} ({len(src_sentences)} lines)')

ref_path = src_path = os.path.join(flores_base_path, flores_split, ref_lang + '.' + flores_split)
with open(ref_path, 'r') as ref_file:
    ref_sentences = [line.strip() for line in ref_file.readlines()]
    print(f'reading {ref_path} ({len(ref_sentences)} lines)')

assert len(ref_sentences) == len(src_sentences)

# constructing and saving data
iso_src = Lang(src_lang.split('_')[0])
iso_ref = Lang(ref_lang.split('_')[0])

out_filename = f"{output_prefix}_{flores_split}_{src_lang.split('_')[0]}-{ref_lang.split('_')[0]}.jsonl"
out_path = os.path.join(output_dir, out_filename)
if os.path.exists(out_path):
    print(81*'=')
    print(f'WARNING: {out_path} already exists!')
    print(81*'=')
    print('Aborting dataset generation.')
else:
    with open(out_path, 'w') as out_file:
        out_template = template.replace('[src_lang]', iso_src.name).replace('[ref_lang]', iso_ref.name)
        for sentence_index in range(len(src_sentences)):
            out_line = {}
            out_line['text'] = out_template.replace('[src_sentence]', src_sentences[sentence_index]).replace('[ref_sentence]', ref_sentences[sentence_index])
            out_file.write(json.dumps(out_line) + '\n')
    print(f'Completed. Out file: {out_path}')




