python3 seq2seq.py --gpu=1 --epoch=80 --unit=256 --layer=3 \
-o="direct_data/join/2" --seed=1 --l2=0.000001 \
data6/issu_diff.train \
data6/msg.train \
data6/diff_issu.vocab \
data6/msg.vocab \
--validation-source="data6/issu_diff.valid" \
--validation-target="data6/msg.valid" \
--test-source="data6/issu_diff.test" \
--test-target="data6/msg.test" \
--max-source-sentence=500 \
--validation-interval=200 \
--max-target-sentence=300  
~/slack_noti.sh "end seq2seq"
