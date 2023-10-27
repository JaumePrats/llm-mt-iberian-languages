from sacrebleu.metrics import BLEU, CHRF, TER
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