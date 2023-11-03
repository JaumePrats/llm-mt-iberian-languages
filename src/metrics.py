from sacrebleu.metrics import BLEU, CHRF, TER
from iso639 import Lang
from langid.langid import LanguageIdentifier, model
import subprocess
import torch
import os

def bleu_score(tgt_path, ref_path):
    '''
    Computes and returns BLEU score and signature for the given target and reference files.

    :param tgt_path: str
        Path to the target file that contains the translations produced by the system to evaluate.
    :param ref_path: str
        Path to the file that contains the references.
    :return: pair (score, signature)
        score: BLEUScore
            Contains the results of BLEU computattion. The score can be accessed via .score .
        signature: BLEUSignature
            Reproducability signature for the BLEU score.
    '''
    bleu = BLEU()

    with open(ref_path, 'r') as file:
        ref = [line.rstrip() for line in file]
    ref_void = [None] * len(ref)
    refs = [ref, ref_void]

    with open(tgt_path, 'r') as file:
        sys = [line.rstrip() for line in file]

    score = bleu.corpus_score(sys, refs)
    return score.format(width = 2), bleu.get_signature()

def comet_score(src_path, tgt_path, ref_path, model = "Unbabel/wmt22-comet-da", score_path = "comet.txt") -> float:
    '''
    Computes and returns COMET score for the given source, target and reference files.
    Model used: Unbabel/wmt22-comet-da

    :param src_path: str
        Path to the source file.
    :param tgt_path: str
        Path to the target file that contains the translations produced by the system to evaluate.
    :param ref_path: str
        Path to the file that contains the references.
    :param score_path: str
        Path of the file where the score is saved.
    :return: float
        value of the COMET score multiplied by 100, ranging from 0 to 100.
    '''

    def read_comet():
        with open(score_path, 'r') as f:
            comet = list(f.readlines())
            assert len(comet) == 1, score_path
            return float(comet[0].split("\t")[1].replace("score:", "").strip()) * 100

    if os.path.isfile(score_path):
        print(f"Comet already exists, reading from {str(score_path)}")
        return read_comet()

    cmd = [
        "comet-score",
        "-s", src_path,
        "-t", tgt_path,
        "-r", ref_path,
        "--model", model,
        "--quiet",
        "--only_system"
    ]
    if not torch.cuda.is_available():
        cmd.extend(["--gpus", "0"])
    cmd.extend([
        ">", str(score_path)
    ])
    subprocess.call(" ".join(cmd), shell=True)
    return read_comet()

def off_target_score(tgt_path, ref_lang, return_tgt_langs_stats=False, max_stat_lines = 10):
    '''
    Computes and returns the percentage (score) of off-target translation in the target file.
    If return_tgt_langs_stats=True, returns the score and a dictionary with the language identification statistics of the target file.

    :param tgt_path: str
        Path to the tgt file.
    :param ref_lang: str
        Code of the reference language, example: 'es' or 'spa'.
    :param return_tgt_langs_stats: bool
        If True, a dictionary with the language identification statistics of the tgt file is also returned.
    :param max_stat_lines: int
        Max number of lines to include in the tgt_langs_stats[lines] for each language.
    :return:
        score: float
            Percentage of off-target translation in tgt file.
        tgt_langs_stats: dict (only if return_tgt_langs_stats=True)
            Dictionary with the statistics of languages present in target file.
    '''
    iso_lg = Lang(ref_lang)
    ref_lang = iso_lg.pt1 # convert ref_lang into iso code

    lang_identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)

    with open(tgt_path, 'r') as tgt_file:
        tgt_lines = [line.strip() for line in tgt_file.readlines()]
    lang_probs = [lang_identifier.classify(line) for line in tgt_lines]

    tgt_langs_lines = {} # for each language the lines on which it appears
    for i, lang_prob in enumerate(lang_probs):
        line_number = i + 1
        lang, prob = lang_prob
        if lang in tgt_langs_lines:
            tgt_langs_lines[lang].append(line_number)
        else:
            tgt_langs_lines[lang] = [line_number]

    if ref_lang in tgt_langs_lines:
        score = round(((len(tgt_lines) - len(tgt_langs_lines[ref_lang])) / len(tgt_lines)) * 100, 4)
    else:
        score = 100

    if return_tgt_langs_stats:
        tgt_langs_stats = {}
        for lang in tgt_langs_lines:
            lang_stats = {}
            iso_lg = Lang(lang)
            lang_stats['name'] = iso_lg.name
            lang_stats['abs_count'] = len(tgt_langs_lines[lang])
            lang_stats['percentage(%)'] = round((lang_stats['abs_count'] / len(tgt_lines)) * 100, 4)
            if lang_stats['abs_count'] < max_stat_lines:
                lang_stats['lines'] = tgt_langs_lines[lang]
            else:
                lang_stats['lines'] = f'More than {max_stat_lines} lines'
            tgt_langs_stats[lang] = lang_stats
        return score, tgt_langs_stats
    else:
        return score
    
    