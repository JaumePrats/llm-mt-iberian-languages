import os
from metrics import off_target_score
import json
from tqdm import tqdm

RESULTS_DIR = "/fs/alvis0/jprats/code/llm-mt-iberian-languages/results"
TGT_DIR = "/fs/alvis0/jprats/code/llm-mt-iberian-languages/tgt_out"
ref_lang = 'spa'

for filename in tqdm(os.listdir(RESULTS_DIR)):
    file_path = os.path.join(RESULTS_DIR, filename)
    if os.path.isfile(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        tgt_path = os.path.join(TGT_DIR, filename.split('txt')[0] + 'spa')
        try:
            ot_score, ot_stats = off_target_score(tgt_path, ref_lang, return_tgt_langs_stats=True)
            lines[30] = f"\nOFF-TARGET TRANSLATION: {10*'-'}\nOFF-TGT(%) = {ot_score}\n" + json.dumps(ot_stats, sort_keys=True, indent=4) + '\n\n'
            if len(lines) == 41:
                lines[39] = f"\ncopy results (blue comet22 comet20 off-tgt) {10*'-'}\n"
                lines[40] = lines[40].strip() + f' {ot_score}\n'
            with open(file_path, 'w') as f:
                f.writelines(lines)
        except Exception as e:
            print(e)
        


    