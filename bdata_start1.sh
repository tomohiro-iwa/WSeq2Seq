
python3 WSeq2Seq.py --gpu=0 --epoch=80 --unit=512 --layer=3 \
-o="all_bdata/wseq/0" --save="save" --seed=0 --l2=0.000001 \
datab/issu.train \
datab/diff.train \
datab/msg.train \
datab/issu.vocab \
datab/diff.vocab \
datab/msg.vocab \
--validation-source0="datab/issu.valid" \
--validation-source1="datab/diff.valid" \
--validation-target="datab/msg.valid" \
--test-source0="datab/issu.test" \
--test-source1="datab/diff.test" \
--test-target="datab/msg.test"


python3 WSeq2Seq.py --gpu=0 --epoch=80 --unit=512 --layer=3 \
-o="all_bdata/wseq/1" --save="save" --seed=1 --l2=0.000001 \
datab/issu.train \
datab/diff.train \
datab/msg.train \
datab/issu.vocab \
datab/diff.vocab \
datab/msg.vocab \
--validation-source0="datab/issu.valid" \
--validation-source1="datab/diff.valid" \
--validation-target="datab/msg.valid" \
--test-source0="datab/issu.test" \
--test-source1="datab/diff.test" \
--test-target="datab/msg.test"


python3 WSeq2Seq.py --gpu=0 --epoch=80 --unit=512 --layer=3 \
-o="all_bdata/wseq/2" --save="save" --seed=2 --l2=0.000001 \
datab/issu.train \
datab/diff.train \
datab/msg.train \
datab/issu.vocab \
datab/diff.vocab \
datab/msg.vocab \
--validation-source0="datab/issu.valid" \
--validation-source1="datab/diff.valid" \
--validation-target="datab/msg.valid" \
--test-source0="datab/issu.test" \
--test-source1="datab/diff.test" \
--test-target="datab/msg.test"


python3 WSeq2Seq.py --gpu=0 --epoch=80 --unit=512 --layer=3 \
-o="all_bdata/wseq/3" --save="save" --seed=3 --l2=0.000001 \
datab/issu.train \
datab/diff.train \
datab/msg.train \
datab/issu.vocab \
datab/diff.vocab \
datab/msg.vocab \
--validation-source0="datab/issu.valid" \
--validation-source1="datab/diff.valid" \
--validation-target="datab/msg.valid" \
--test-source0="datab/issu.test" \
--test-source1="datab/diff.test" \
--test-target="datab/msg.test"


python3 WSeq2Seq.py --gpu=0 --epoch=80 --unit=512 --layer=3 \
-o="all_bdata/wseq/4" --save="save" --seed=4 --l2=0.000001 \
datab/issu.train \
datab/diff.train \
datab/msg.train \
datab/issu.vocab \
datab/diff.vocab \
datab/msg.vocab \
--validation-source0="datab/issu.valid" \
--validation-source1="datab/diff.valid" \
--validation-target="datab/msg.valid" \
--test-source0="datab/issu.test" \
--test-source1="datab/diff.test" \
--test-target="datab/msg.test"

