import os
from metrics import off_target_score
import json
from tqdm import tqdm

def add_offtgt(ref_lang, results_dir, tgt_dir):
    for filename in tqdm(os.listdir(results_dir)):
        file_path = os.path.join(results_dir, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as f:
                lines = f.readlines()
            tgt_path = os.path.join(tgt_dir, filename.split('txt')[0] + 'spa')
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
        
# RESULTS_DIR = "/fs/alvis0/jprats/code/llm-mt-iberian-languages/results"
# TGT_DIR = "/fs/alvis0/jprats/code/llm-mt-iberian-languages/tgt_out"
# ref_lang = 'spa'
# add_offtgt(ref_lang=ref_lang, results_dir=RESULTS_DIR, tgt_dir=TGT_DIR) 



########################################################################
#
# Test off-target metric on evaluation reference file to see sensitivity:
#
########################################################################
ref_path = '/fs/surtr0/jprats/data/processed/evaluation/UNPC/testset/UNv1.0.testset.spa'
ref_lang = 'spa'
print(20*'=')
print('reference_file:', ref_path)
print('reference language:', ref_lang)
print(20*'-')
ot_score, ot_stats = off_target_score(tgt_path=ref_path, ref_lang=ref_lang, return_tgt_langs_stats=True)
print(f"OFF-TGT(%) = {ot_score}")
print(json.dumps(ot_stats, sort_keys=True, indent=4))
