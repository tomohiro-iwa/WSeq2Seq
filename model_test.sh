python3 WSeq2Seq_test.py --epoch=80 --unit=256 --layer=3 \
-o="wresult/20" --save="save" --seed=1 --l2=0.000001 \
data5/issu.train \
data5/diff.train \
data5/msg.train \
data5/issu.vocab \
data5/diff.vocab \
data5/msg.vocab \
--validation-source0="data5/issu.valid" \
--validation-source1="data5/diff.valid" \
--validation-target="data5/msg.valid" \
--test-source0="data5/issu.test" \
--test-source1="data5/diff.test" \
--test-target="data5/msg.test" 
~/slack_noti.sh "end Wseq2seq"
