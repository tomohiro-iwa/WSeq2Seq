
python3 WSeq2Seq.py --gpu=1 --epoch=80 --unit=512 --layer=3 \
-o="all_data5_pp/wseq/0" --save="save" --seed=0 --l2=0.000001 \
data5_pp/issu.train \
data5_pp/diff.train \
data5_pp/msg.train \
data5_pp/issu.vocab \
data5_pp/diff.vocab \
data5_pp/msg.vocab \
--validation-source0="data5_pp/issu.valid" \
--validation-source1="data5_pp/diff.valid" \
--validation-target="data5_pp/msg.valid" \
--test-source0="data5_pp/issu.test" \
--test-source1="data5_pp/diff.test" \
--test-target="data5_pp/msg.test"


python3 WSeq2Seq.py --gpu=1 --epoch=80 --unit=512 --layer=3 \
-o="all_data5_pp/wseq/1" --save="save" --seed=1 --l2=0.000001 \
data5_pp/issu.train \
data5_pp/diff.train \
data5_pp/msg.train \
data5_pp/issu.vocab \
data5_pp/diff.vocab \
data5_pp/msg.vocab \
--validation-source0="data5_pp/issu.valid" \
--validation-source1="data5_pp/diff.valid" \
--validation-target="data5_pp/msg.valid" \
--test-source0="data5_pp/issu.test" \
--test-source1="data5_pp/diff.test" \
--test-target="data5_pp/msg.test"


python3 WSeq2Seq.py --gpu=1 --epoch=80 --unit=512 --layer=3 \
-o="all_data5_pp/wseq/2" --save="save" --seed=2 --l2=0.000001 \
data5_pp/issu.train \
data5_pp/diff.train \
data5_pp/msg.train \
data5_pp/issu.vocab \
data5_pp/diff.vocab \
data5_pp/msg.vocab \
--validation-source0="data5_pp/issu.valid" \
--validation-source1="data5_pp/diff.valid" \
--validation-target="data5_pp/msg.valid" \
--test-source0="data5_pp/issu.test" \
--test-source1="data5_pp/diff.test" \
--test-target="data5_pp/msg.test"


python3 WSeq2Seq.py --gpu=1 --epoch=80 --unit=512 --layer=3 \
-o="all_data5_pp/wseq/3" --save="save" --seed=3 --l2=0.000001 \
data5_pp/issu.train \
data5_pp/diff.train \
data5_pp/msg.train \
data5_pp/issu.vocab \
data5_pp/diff.vocab \
data5_pp/msg.vocab \
--validation-source0="data5_pp/issu.valid" \
--validation-source1="data5_pp/diff.valid" \
--validation-target="data5_pp/msg.valid" \
--test-source0="data5_pp/issu.test" \
--test-source1="data5_pp/diff.test" \
--test-target="data5_pp/msg.test"


python3 WSeq2Seq.py --gpu=1 --epoch=80 --unit=512 --layer=3 \
-o="all_data5_pp/wseq/4" --save="save" --seed=4 --l2=0.000001 \
data5_pp/issu.train \
data5_pp/diff.train \
data5_pp/msg.train \
data5_pp/issu.vocab \
data5_pp/diff.vocab \
data5_pp/msg.vocab \
--validation-source0="data5_pp/issu.valid" \
--validation-source1="data5_pp/diff.valid" \
--validation-target="data5_pp/msg.valid" \
--test-source0="data5_pp/issu.test" \
--test-source1="data5_pp/diff.test" \
--test-target="data5_pp/msg.test"

