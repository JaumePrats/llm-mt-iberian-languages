from metrics import bleu_score, comet_score, off_target_score
import argparse
import os
from tqdm import tqdm
import json

TGT_OUT_DIR = '/fs/surtr0/jprats/code/llm-mt-iberian-languages/tgt_out'
COMPLETE_OUT_DIR = '/fs/surtr0/jprats/code/llm-mt-iberian-languages/complete_out'
COMET22_DIR = '/fs/surtr0/jprats/code/llm-mt-iberian-languages/results/mt_eval/comet_scores/comet22'
COMET20_DIR = '/fs/surtr0/jprats/code/llm-mt-iberian-languages/results/mt_eval/comet_scores/comet20'

BOS = '<s>'
EOS = '</s>'

def reprocess_output_file(complete_out_file_path, tgt_lang):
    new_filename = os.path.splitext(os.path.basename(complete_out_file_path))[0] + f'_REPROCESSED.{tgt_lang}'
    new_tgt_out_file_path = os.path.join(TGT_OUT_DIR, new_filename)

    num_fewshot = int(complete_out_file_path.split('nshot')[1].split('_')[0])

    with open(complete_out_file_path, 'r') as complete_out_file:
        data = complete_out_file.read()
        outputs = data.split('--------------------\n')

    tgt_sentences = []
    for output in outputs:
        # import pdb; pdb.set_trace()
        if output != '':
            raw_tgt_sentence = output.split(BOS)[(num_fewshot+1)*2]
            if len(raw_tgt_sentence.split(EOS)) > 1: # template eos found in model response
                tgt_sentence = raw_tgt_sentence.split(EOS)[0]
                tgt_sentence = tgt_sentence.replace('\n', ' ') # making sure that output does not contain \n
            else: # if template eos not found, using breakline as eos
                tgt_sentence = raw_tgt_sentence.split('\n')[0]
            tgt_sentences.append(tgt_sentence)

    with open(new_tgt_out_file_path, 'w') as new_tgt_out_file:
        for tgt_sentence in tgt_sentences:
            new_tgt_out_file.write(tgt_sentence + '\n')
    print(f'CREATED: {new_tgt_out_file_path}')
    return new_tgt_out_file_path

def reevaluate(src_path, tgt_path, ref_path, ref_lang_code, results_path):
    comet_filename = os.path.splitext(os.path.basename(results_path))[0]

    with open(results_path, 'a') as results_file:
        results_file.write('\n')
        results_file.write(80*'*' + '\n')
        results_file.write('*\n')
        results_file.write('*\tREEVALUATION RESULTS\n')
        results_file.write('*\n')
        results_file.write(80*'*' + '\n')
        results_file.write('\n')

        # OFF-TARGET TRANSLATION
        results_file.write(f"\nOFF-TARGET TRANSLATION: {10*'-'}\n")
        ot_score, ot_stats = off_target_score(tgt_path, ref_lang_code, return_tgt_langs_stats=True)
        results_file.write('OFF-TGT(%) = '+ str(ot_score) + '\n')
        results_file.write(json.dumps(ot_stats, sort_keys=True, indent=4) + '\n')

        # BLEU
        results_file.write(f"\nBLEU: {10*'-'}\n")
        b_score, b_signature = bleu_score(tgt_path, ref_path)
        results_file.write(b_score + '\n')
        results_file.write('Signature: '+ str(b_signature) + '\n')

        # COMET
        results_file.write(f"\nCOMET: {10*'-'}\n")
        # comet22
        comet22_score_path = os.path.join(COMET22_DIR, comet_filename)
        c22_score = comet_score(src_path, tgt_path, ref_path, model="Unbabel/wmt22-comet-da", score_path=comet22_score_path)
        results_file.write('COMET22 = '+ str(c22_score) + '\n') 
        #comet20
        comet20_score_path = os.path.join(COMET20_DIR, comet_filename)
        c20_score = comet_score(src_path, tgt_path, ref_path, model="Unbabel/wmt20-comet-da", score_path=comet20_score_path)
        results_file.write('COMET20 = '+ str(c20_score) + '\n')

        #add line to copy in spreadsheet
        results_file.write(f"\ncopy results (blue comet22 comet20 off-tgt) {10*'-'}\n")
        results_file.write(f"{b_score.split(' ')[2]} {c22_score} {c20_score} {ot_score}\n")
        print(f'UPDATED: {results_path}')


def main(file_prefix, results_dir, data_dir):

    for filename in tqdm(os.listdir(results_dir)):
        if filename.startswith(file_prefix):
            complete_out_file_path = os.path.join(COMPLETE_OUT_DIR, filename)
            results_file_path = os.path.join(results_dir, filename)

            # obtaining tgt_lang
            file_basename = os.path.splitext(os.path.basename(complete_out_file_path))[0]
            old_tgt_out_name = [fname for fname in os.listdir(TGT_OUT_DIR) if fname.startswith(file_basename)]
            assert len(old_tgt_out_name) == 1          
            ref_lang = os.path.splitext(os.path.basename(old_tgt_out_name[0]))[1].split('.')[1]

            # obtaining src_lang
            src_lang = file_basename.split(f'-{ref_lang}')[0][-3:]

            # obtaining src and tgt files
            for dataset_file in os.listdir(data_dir):
                if ref_lang in dataset_file:
                    ref_file_path = os.path.join(data_dir, dataset_file)
                if src_lang in dataset_file:
                    src_file_path = os.path.join(data_dir, dataset_file)
            assert src_file_path
            assert ref_file_path

            print(f"\nAssuming that template used has bos: '{BOS}' and eos: '{EOS}'")
            # import pdb; pdb.set_trace()
            new_tgt_out_path = reprocess_output_file(complete_out_file_path, ref_lang)
            reevaluate(
                src_path=src_file_path,
                tgt_path=new_tgt_out_path,
                ref_path=ref_file_path,
                ref_lang_code=ref_lang,
                results_path=results_file_path
            )





                    



            





if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Re-process model outputs of a selection of files and recompute metrics.')
    parser.add_argument('--file-prefix', type=str, required=True, help='prefix of the files to process')
    parser.add_argument('--results-dir',  type=str, required=True,  help='directory with the results file to process.')
    parser.add_argument('--data-dir',  type=str, required=True,  help='directory that contains dataset files (src and ref) to compute MT metrics.')

    params=parser.parse_args()

    main(params.file_prefix, params.results_dir, params.data_dir)