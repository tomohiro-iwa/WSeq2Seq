python3 WSeq2Seq.py --gpu=1 --epoch=80 --unit=512 --layer=3 \
-o="direct_data/wseq/1" --save="save" --seed=1 --l2=0.000001 \
data6/issu.train \
data6/diff.train \
data6/msg.train \
data6/issu.vocab \
data6/diff.vocab \
data6/msg.vocab \
--validation-source0="data6/issu.valid" \
--validation-source1="data6/diff.valid" \
--validation-target="data6/msg.valid" \
--test-source0="data6/issu.test" \
--test-source1="data6/diff.test" \
--test-target="data6/msg.test" 
~/slack_noti.sh "end Wseq2seq"
