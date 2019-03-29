python3 seq2seq.py --gpu=0 --epoch=80 --unit=256 --layer=3 \
-o="direct_data/join/1" --seed=1 --l2=0.000001 \
data6/diff_issu.train \
data6/msg.train \
data6/diff_issu.vocab \
data6/msg.vocab \
--validation-source="data6/diff_issu.valid" \
--validation-target="data6/msg.valid" \
--test-source="data6/diff_issu.test" \
--test-target="data6/msg.test" \
--max-source-sentence=500 \
--validation-interval=200 \
--max-target-sentence=300  
~/slack_noti.sh "end seq2seq"
