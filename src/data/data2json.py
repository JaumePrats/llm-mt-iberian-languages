import json
import os
from iso639 import Lang

# IN
data_path = '/fs/surtr0/jprats/data/raw/UNPC/data'
src_lang = 'en'
ref_lang = 'es'
# instruction_template = "###SRC"
# response_template = "###TGT"
instruction_template = ""
response_template = ""
template = f"{instruction_template} [src_lang]: <s>[src_sentence]</s>\n{response_template} [ref_lang]: <s>[ref_sentence]</s>"

# OUT
output_dir = '/fs/surtr0/jprats/data/processed/parallel_ft/UNPC'
output_prefix = 'UNPC'
out_filename = f"{output_prefix}_{src_lang.split('_')[0]}-{ref_lang.split('_')[0]}.jsonl"

# -----------------------

# reading input files
for filename in os.listdir(data_path):
    file_extension = os.path.splitext(filename)[1]
    if file_extension == f'.{src_lang}':
        src_path = os.path.join(data_path, filename)
    if file_extension == f'.{ref_lang}':
        ref_path = os.path.join(data_path, filename)

with open(src_path, 'r') as src_file:
    src_sentences = [line.strip() for line in src_file.readlines()]
    print(f'reading {src_path} ({len(src_sentences)} lines)')

with open(ref_path, 'r') as ref_file:
    ref_sentences = [line.strip() for line in ref_file.readlines()]
    print(f'reading {ref_path} ({len(ref_sentences)} lines)')

assert len(ref_sentences) == len(src_sentences)

# constructing and saving data
iso_src = Lang(src_lang)
iso_ref = Lang(ref_lang)

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




