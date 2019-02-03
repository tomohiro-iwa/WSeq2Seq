
python3 WSeq2Seq.py --gpu=1 --epoch=80 --unit=512 --layer=2 \
-o="wdata/35" --save="save" --seed=0 --l2=0.000001 \
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


python3 WSeq2Seq.py --gpu=1 --epoch=80 --unit=512 --layer=2 \
-o="wdata/36" --save="save" --seed=1 --l2=0.000001 \
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


python3 WSeq2Seq.py --gpu=1 --epoch=80 --unit=512 --layer=2 \
-o="wdata/37" --save="save" --seed=2 --l2=0.000001 \
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


python3 WSeq2Seq.py --gpu=1 --epoch=80 --unit=512 --layer=2 \
-o="wdata/38" --save="save" --seed=3 --l2=0.000001 \
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


python3 WSeq2Seq.py --gpu=1 --epoch=80 --unit=512 --layer=2 \
-o="wdata/39" --save="save" --seed=4 --l2=0.000001 \
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

