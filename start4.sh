python3 WSeq2Seq.py --gpu=0 --epoch=40 \
data3/issu3.train.txt \
data3/diff3.train.txt \
data3/msg3.train.txt \
data/issu.vocab \
data/diff.vocab \
data/msg.vocab \
--validation-source0="data3/issu3.valid.txt" \
--validation-source1="data3/diff3.valid.txt" \
--validation-target="data3/msg3.valid.txt" \
--test-source0="data3/issu3.test.txt" \
--test-source1="data3/diff3.test.txt" \
--test-target="data3/msg3.test.txt" 
~/slack_noti.sh "end Wseq2seq"
