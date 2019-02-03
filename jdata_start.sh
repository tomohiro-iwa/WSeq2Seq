
python3 seq2seq.py --gpu=0 --epoch=120 --unit=256 --layer=3 \
-o="jdata/25" --seed=0 --l2=0.000001 \
data5/join2.train \
data5/msg.train \
data5/join.vocab \
data5/msg.vocab \
--validation-source="data5/join2.valid" \
--validation-target="data5/msg.valid" \
--test-source="data5/join2.test" \
--test-target="data5/msg.test" \
--max-source-sentence=500 \
--validation-interval=200 \
--max-target-sentence=300  


python3 seq2seq.py --gpu=0 --epoch=120 --unit=256 --layer=3 \
-o="jdata/26" --seed=1 --l2=0.000001 \
data5/join2.train \
data5/msg.train \
data5/join.vocab \
data5/msg.vocab \
--validation-source="data5/join2.valid" \
--validation-target="data5/msg.valid" \
--test-source="data5/join2.test" \
--test-target="data5/msg.test" \
--max-source-sentence=500 \
--validation-interval=200 \
--max-target-sentence=300  


python3 seq2seq.py --gpu=0 --epoch=120 --unit=256 --layer=3 \
-o="jdata/27" --seed=2 --l2=0.000001 \
data5/join2.train \
data5/msg.train \
data5/join.vocab \
data5/msg.vocab \
--validation-source="data5/join2.valid" \
--validation-target="data5/msg.valid" \
--test-source="data5/join2.test" \
--test-target="data5/msg.test" \
--max-source-sentence=500 \
--validation-interval=200 \
--max-target-sentence=300  


~/slack_noti.sh "end of seq2seq about jdata1"
