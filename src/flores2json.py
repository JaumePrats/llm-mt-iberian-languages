import json
import os

# IN
flores_base_path = '/fs/surtr0/jprats/data/raw/flores200_dataset'
flores_split = 'dev'
languages = [
    'eng_Latn',
    'spa_Latn',
    'cat_Latn'
]

# OUT
output_dir = '/fs/surtr0/jprats/data/processed'
output_prefix = 'flores_text'

# -----------------------

src_sentences = [] # list of lists dim: n_langs * number of src sentences
for lang in languages:
    src_path = src_path = os.path.join(flores_base_path, flores_split, lang + '.' + flores_split)
    with open(src_path, 'r') as src_file:
        src_lines = [line.strip() for line in src_file.readlines()]
    
    # make sure that all files have the same number of lines
    new_src_length = len(src_lines)
    print(f'reading {src_path} ({new_src_length} lines)')
    print('...')
    if 'src_length' in locals():
        assert new_src_length == src_length
    src_length = new_src_length

    src_sentences.append(src_lines)

out_filename = f'{output_prefix}_{flores_split}_'
for i, lang in enumerate(languages):
    if i == 0:
        out_filename = out_filename + lang.split('_')[0]
    else:
        out_filename = out_filename + '-' + lang.split('_')[0]
out_filename = out_filename + '.jsonl'

out_path = os.path.join(output_dir, out_filename)
with open(out_path, 'w') as out_file:
    for sentence_index in range(len(src_sentences[0])):
        # out_line = {}
        # for lang_index, lang in enumerate(languages):
        #     lang_code = lang.split('_')[0]
        #     out_line[lang_code] = src_sentences[lang_index][sentence_index]
        # out_file.write(json.dumps(out_line) + "\n")
        out_line = {}
        out_line['text'] = f"### English: <s>{src_sentences[0][sentence_index]}</s>\n ### Spanish: <s>{src_sentences[1][sentence_index]}</s>\n"
        out_file.write(json.dumps(out_line) + "\n")

print(f'Completed. Out file: {out_path}')





