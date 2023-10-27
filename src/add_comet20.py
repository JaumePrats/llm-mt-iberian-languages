import os
from metrics import comet_score

RESULTS_DIR = "/fs/alvis0/jprats/code/llm-mt-iberian-languages/results"
TGT_DIR = "/fs/alvis0/jprats/code/llm-mt-iberian-languages/tgt_out"
src_path = "/fs/surtr0/jprats/data/raw/flores200_dataset/dev/eng_Latn.dev"
ref_path = "/fs/surtr0/jprats/data/raw/flores200_dataset/dev/spa_Latn.dev"
COMET20_DIR = "/fs/alvis0/jprats/code/llm-mt-iberian-languages/results/comet_scores/comet20"

for filename in os.listdir(RESULTS_DIR):
    file_path = os.path.join(RESULTS_DIR, filename)
    if os.path.isfile(file_path) and filename.startswith('falcon_flores_eng-spa'):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        if len(lines) == 37: # only Comet22
            tgt_path = os.path.join(TGT_DIR, filename.split('txt')[0] + 'spa')
            score_path = os.path.join(COMET20_DIR, filename.split('.txt')[0])
            c20_score = comet_score(src_path=src_path, tgt_path=tgt_path, ref_path=ref_path, model="Unbabel/wmt20-comet-da", score_path=score_path)
            with open(file_path, 'a') as f:
                f.write('COMET20 = '+ str(c20_score) + '\n')
            #break


    