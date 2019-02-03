
python3 WSeq2Seq.py --gpu=0 --epoch=80 --unit=512 --layer=3 \
-o="wgdata/0" --save="save" --seed=0 --l2=0.000001 \
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


python3 WSeq2Seq.py --gpu=0 --epoch=80 --unit=512 --layer=3 \
-o="wgdata/1" --save="save" --seed=1 --l2=0.000001 \
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


python3 WSeq2Seq.py --gpu=0 --epoch=80 --unit=512 --layer=3 \
-o="wgdata/2" --save="save" --seed=2 --l2=0.000001 \
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


python3 WSeq2Seq.py --gpu=0 --epoch=80 --unit=512 --layer=3 \
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


python3 WSeq2Seq.py --gpu=0 --epoch=80 --unit=512 --layer=3 \
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

