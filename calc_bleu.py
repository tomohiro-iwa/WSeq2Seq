from nltk.translate import bleu_score
result_id = 2
expect_id = 3
all_id = 5

i = 0
for line in open("hoge.txt"):
	if i%all_id == result_id:
		result = line[11:]

	if i%all_id == expect_id:
		expect = line[11:]
		bleu = bleu_score.corpus_bleu(
		    [result], [expect],
		    smoothing_function=bleu_score.SmoothingFunction().method1)
		print(bleu)
	i += 1
