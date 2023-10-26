from sacrebleu.metrics import BLEU, CHRF, TER

bleu = BLEU()

with open('/fs/surtr0/jprats/data/raw/flores200_dataset/dev/spa_Latn.dev', 'r') as file:
    ref = [line.rstrip() for line in file]
ref_void = [None] * len(ref)
refs = [ref, ref_void]

with open('/fs/alvis0/jprats/code/falcon_test_flores_tr.es', 'r') as file:
    sys = [line.rstrip() for line in file]

score = bleu.corpus_score(sys, refs)
print(score.format(width=3))
print(score.score)

print(bleu.get_signature())