python3 WSeq2Seq.py --gpu=0 --epoch=40 --unit=256 --layer=2 -o="result/basic" \
basic/j1.train \
basic/j2.train \
basic/ee.train \
basic/j1-vocab.txt \
basic/j2-vocab.txt \
basic/ee-vocab.txt \
--validation-source0="basic/j1.valid" \
--validation-source1="basic/j2.valid" \
--validation-target="basic/ee.valid" \
--test-source0="basic/j1.test" \
--test-source1="basic/j2.test" \
--test-target="basic/ee.test" 
~/slack_noti.sh "end Wseq2seq"
