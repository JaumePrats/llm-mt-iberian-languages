import os
from metrics import comet_score

RESULTS_DIR = "/fs/surtr0/jprats/code/llm-mt-iberian-languages/results"
TGT_DIR = "/fs/surtr0/jprats/code/llm-mt-iberian-languages/tgt_out"
SRC_DIR = "/fs/surtr0/jprats/data/raw/flores200_dataset/dev"
COMET20_DIR = "/fs/surtr0/jprats/code/llm-mt-iberian-languages/results/comet_scores/comet20"
COMET22_DIR = "/fs/surtr0/jprats/code/llm-mt-iberian-languages/results/comet_scores/comet22"

for filename in os.listdir(RESULTS_DIR):
    file_path = os.path.join(RESULTS_DIR, filename)
    if os.path.isfile(file_path) and filename.startswith('EVAL_falcon_flores'):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        if lines[-1].startswith('COMET: ----------'): # Crashed while computing comet

            print(f'Adding comet scores to {file_path}')
            # get translatation direction
            tr_direction = lines[3].strip().split(' ')[2] # eng-cat
            src_lang = tr_direction.split('-')[0]
            tgt_lang = tr_direction.split('-')[1]

            src_path = os.path.join(SRC_DIR, src_lang + '_Latn.dev')
            ref_path = os.path.join(SRC_DIR, tgt_lang + '_Latn.dev')
            tgt_path = os.path.join(TGT_DIR, filename.split('txt')[0] + tgt_lang)

            c22_path = os.path.join(COMET22_DIR, filename.split('.txt')[0])
            c22_score = comet_score(src_path=src_path, tgt_path=tgt_path, ref_path=ref_path, model="Unbabel/wmt22-comet-da", score_path=c22_path)

            c20_path = os.path.join(COMET20_DIR, filename.split('.txt')[0])
            c20_score = comet_score(src_path=src_path, tgt_path=tgt_path, ref_path=ref_path, model="Unbabel/wmt20-comet-da", score_path=c20_path)

            with open(file_path, 'a') as f:
                f.write('COMET22 = '+ str(c22_score) + '\n')
                f.write('COMET20 = '+ str(c20_score) + '\n')

        # if len(lines) == 37: # only Comet22
        #     tgt_path = os.path.join(TGT_DIR, filename.split('txt')[0] + 'spa')
        #     score_path = os.path.join(COMET20_DIR, filename.split('.txt')[0])
        #     c20_score = comet_score(src_path=src_path, tgt_path=tgt_path, ref_path=ref_path, model="Unbabel/wmt20-comet-da", score_path=score_path)
        #     with open(file_path, 'a') as f:
        #         f.write('COMET20 = '+ str(c20_score) + '\n')
            #break


    