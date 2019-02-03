python3 WSeq2Seq.py --gpu=0 --unit=512 \
-o="wresult/20" --save="save" --seed=1 --l2=0.000001 \
wseq_verif/prep.e1.train \
wseq_verif/prep.e2.train \
wseq_verif/ee.train \
wseq_verif/e1.vocab \
wseq_verif/e2.vocab \
wseq_verif/vocab.fr \
--validation-source0="wseq_verif/prep.e1.valid" \
--validation-source1="wseq_verif/prep.e2.valid" \
--validation-target="wseq_verif/ee.valid" \
--test-source0="wseq_verif/prep.e1.test" \
--test-source1="wseq_verif/prep.e2.test" \
--test-target="wseq_verif/ee.test" 
~/slack_noti.sh "end Wseq2seq"
