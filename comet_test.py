import subprocess
import torch
import os

def comet_score(src_path, tgt_path, ref_path) -> float:
    score_path = "comet.txt"

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
        "--model", "Unbabel/wmt22-comet-da",
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

score = comet_score(
    src_path='/fs/surtr0/jprats/data/raw/flores200_dataset/dev/eng_Latn.dev',
    tgt_path='/fs/alvis0/jprats/code/falcon_test_flores_tr.es',
    ref_path='/fs/surtr0/jprats/data/raw/flores200_dataset/dev/spa_Latn.dev'
    )
print('The score is: ', str(score))