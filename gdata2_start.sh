
python3 WSeq2Seq.py --gpu=1 --epoch=80 --unit=512 --layer=3 \
-o="wgdata/3" --save="save" --seed=3 --l2=0.000001 \
datag/issu.train \
datag/diff.train \
datag/msg.train \
datag/issu.vocab \
datag/diff.vocab \
datag/msg.vocab \
--validation-source0="datag/issu.valid" \
--validation-source1="datag/diff.valid" \
--validation-target="datag/msg.valid" \
--test-source0="datag/issu.test" \
--test-source1="datag/diff.test" \
--test-target="datag/msg.test"


python3 WSeq2Seq.py --gpu=1 --epoch=80 --unit=512 --layer=3 \
-o="wgdata/4" --save="save" --seed=4 --l2=0.000001 \
datag/issu.train \
datag/diff.train \
datag/msg.train \
datag/issu.vocab \
datag/diff.vocab \
datag/msg.vocab \
--validation-source0="datag/issu.valid" \
--validation-source1="datag/diff.valid" \
--validation-target="datag/msg.valid" \
--test-source0="datag/issu.test" \
--test-source1="datag/diff.test" \
--test-target="datag/msg.test"


python3 seq2seq.py --gpu=1 --epoch=120 --unit=256 --layer=3 \
-o="sgdata/27" --seed=2 --l2=0.000001 \
datag/diff.train \
datag/msg.train \
datag/diff.vocab \
datag/msg.vocab \
--validation-source="datag/diff.valid" \
--validation-target="datag/msg.valid" \
--test-source="datag/diff.test" \
--test-target="datag/msg.test" \
--max-source-sentence=500 \
--validation-interval=200 \
--max-target-sentence=300  


python3 seq2seq.py --gpu=1 --epoch=120 --unit=256 --layer=3 \
-o="sgdata/28" --seed=3 --l2=0.000001 \
datag/diff.train \
datag/msg.train \
datag/diff.vocab \
datag/msg.vocab \
--validation-source="datag/diff.valid" \
--validation-target="datag/msg.valid" \
--test-source="datag/diff.test" \
--test-target="datag/msg.test" \
--max-source-sentence=500 \
--validation-interval=200 \
--max-target-sentence=300  


python3 seq2seq.py --gpu=1 --epoch=120 --unit=256 --layer=3 \
-o="sgdata/29" --seed=4 --l2=0.000001 \
datag/diff.train \
datag/msg.train \
datag/diff.vocab \
datag/msg.vocab \
--validation-source="datag/diff.valid" \
--validation-target="datag/msg.valid" \
--test-source="datag/diff.test" \
--test-target="datag/msg.test" \
--max-source-sentence=500 \
--validation-interval=200 \
--max-target-sentence=300  

~/slack_noti.sh "end of learn gdata"
