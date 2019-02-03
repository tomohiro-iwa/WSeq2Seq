
python3 WSeq2Seq.py --gpu=0 --epoch=80 --unit=256 --layer=3 \
-o="wdata/20" --save="save" --seed=0 --l2=0.000001 \
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


python3 WSeq2Seq.py --gpu=0 --epoch=80 --unit=256 --layer=3 \
-o="wdata/21" --save="save" --seed=1 --l2=0.000001 \
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


python3 WSeq2Seq.py --gpu=0 --epoch=80 --unit=256 --layer=3 \
-o="wdata/22" --save="save" --seed=2 --l2=0.000001 \
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


python3 WSeq2Seq.py --gpu=0 --epoch=80 --unit=256 --layer=3 \
-o="wdata/23" --save="save" --seed=3 --l2=0.000001 \
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


python3 WSeq2Seq.py --gpu=0 --epoch=80 --unit=256 --layer=3 \
-o="wdata/24" --save="save" --seed=4 --l2=0.000001 \
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


python3 WSeq2Seq.py --gpu=0 --epoch=80 --unit=512 --layer=3 \
-o="wdata/25" --save="save" --seed=0 --l2=0.000001 \
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


python3 WSeq2Seq.py --gpu=0 --epoch=80 --unit=512 --layer=3 \
-o="wdata/26" --save="save" --seed=1 --l2=0.000001 \
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


python3 WSeq2Seq.py --gpu=0 --epoch=80 --unit=512 --layer=3 \
-o="wdata/27" --save="save" --seed=2 --l2=0.000001 \
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


python3 WSeq2Seq.py --gpu=0 --epoch=80 --unit=512 --layer=3 \
-o="wdata/28" --save="save" --seed=3 --l2=0.000001 \
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


python3 WSeq2Seq.py --gpu=0 --epoch=80 --unit=512 --layer=3 \
-o="wdata/29" --save="save" --seed=4 --l2=0.000001 \
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


python3 WSeq2Seq.py --gpu=0 --epoch=80 --unit=512 --layer=2 \
-o="wdata/30" --save="save" --seed=0 --l2=0.000001 \
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


python3 WSeq2Seq.py --gpu=0 --epoch=80 --unit=512 --layer=2 \
-o="wdata/31" --save="save" --seed=1 --l2=0.000001 \
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


python3 WSeq2Seq.py --gpu=0 --epoch=80 --unit=512 --layer=2 \
-o="wdata/32" --save="save" --seed=2 --l2=0.000001 \
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


python3 WSeq2Seq.py --gpu=0 --epoch=80 --unit=512 --layer=2 \
-o="wdata/33" --save="save" --seed=3 --l2=0.000001 \
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


python3 WSeq2Seq.py --gpu=0 --epoch=80 --unit=512 --layer=2 \
-o="wdata/34" --save="save" --seed=4 --l2=0.000001 \
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

