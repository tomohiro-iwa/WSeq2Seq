python3 seq2seq.py --gpu=1 --epoch=80 --unit=1024 --layer=3 -o="sresult/2" --seed=2 \
data4/diff4.train.txt \
data4/msg4.train.txt \
data/diff.vocab \
data/msg.vocab \
--validation-source="data4/diff4.valid.txt" \
--validation-target="data4/msg4.valid.txt" \
--test-source="data4/diff4.valid.txt" \
--test-target="data4/msg4.valid.txt" \
--max-source-sentence=300 \
--validation-interval=200 \
--max-target-sentence=200  
~/slack_noti.sh "end seq2seq"
