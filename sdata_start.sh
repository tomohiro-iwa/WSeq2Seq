
python3 seq2seq.py --gpu=1 --epoch=80 --unit=256 --layer=3 \
-o="sdata/30" --seed=0 --l2=0.000001 \
data5/diff.train \
data5/msg.train \
data5/diff.vocab \
data5/msg.vocab \
--validation-source="data5/diff.valid" \
--validation-target="data5/msg.valid" \
--test-source="data5/diff.test" \
--test-target="data5/msg.test" \
--max-source-sentence=500 \
--validation-interval=200 \
--max-target-sentence=300  


python3 seq2seq.py --gpu=1 --epoch=80 --unit=256 --layer=3 \
-o="sdata/31" --seed=1 --l2=0.000001 \
data5/diff.train \
data5/msg.train \
data5/diff.vocab \
data5/msg.vocab \
--validation-source="data5/diff.valid" \
--validation-target="data5/msg.valid" \
--test-source="data5/diff.test" \
--test-target="data5/msg.test" \
--max-source-sentence=500 \
--validation-interval=200 \
--max-target-sentence=300  


python3 seq2seq.py --gpu=1 --epoch=80 --unit=256 --layer=3 \
-o="sdata/32" --seed=2 --l2=0.000001 \
data5/diff.train \
data5/msg.train \
data5/diff.vocab \
data5/msg.vocab \
--validation-source="data5/diff.valid" \
--validation-target="data5/msg.valid" \
--test-source="data5/diff.test" \
--test-target="data5/msg.test" \
--max-source-sentence=500 \
--validation-interval=200 \
--max-target-sentence=300  


python3 seq2seq.py --gpu=1 --epoch=80 --unit=256 --layer=3 \
-o="sdata/33" --seed=3 --l2=0.000001 \
data5/diff.train \
data5/msg.train \
data5/diff.vocab \
data5/msg.vocab \
--validation-source="data5/diff.valid" \
--validation-target="data5/msg.valid" \
--test-source="data5/diff.test" \
--test-target="data5/msg.test" \
--max-source-sentence=500 \
--validation-interval=200 \
--max-target-sentence=300  


python3 seq2seq.py --gpu=1 --epoch=80 --unit=256 --layer=3 \
-o="sdata/34" --seed=4 --l2=0.000001 \
data5/diff.train \
data5/msg.train \
data5/diff.vocab \
data5/msg.vocab \
--validation-source="data5/diff.valid" \
--validation-target="data5/msg.valid" \
--test-source="data5/diff.test" \
--test-target="data5/msg.test" \
--max-source-sentence=500 \
--validation-interval=200 \
--max-target-sentence=300  

