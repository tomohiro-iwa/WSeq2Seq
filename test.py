from nltk.translate import bleu_score
bleu = bleu_score.corpus_bleu(
    ["this is a pen"], ["this is a pen"],
    )

print(bleu)
