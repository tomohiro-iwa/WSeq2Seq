
python3 seq2seq.py --gpu=1 --epoch=80 --unit=1024 --layer=3 \
-o=" all_bdata/seq/0" --seed=0 --l2=0.000001 \
datab/diff.train \
datab/msg.train \
datab/diff.vocab \
datab/msg.vocab \
--validation-source="datab/diff.valid" \
--validation-target="datab/msg.valid" \
--test-source="datab/diff.test" \
--test-target="datab/msg.test" \
--max-source-sentence=500 \
--validation-interval=200 \
--max-target-sentence=300  


python3 seq2seq.py --gpu=1 --epoch=80 --unit=1024 --layer=3 \
-o=" all_bdata/seq/1" --seed=1 --l2=0.000001 \
datab/diff.train \
datab/msg.train \
datab/diff.vocab \
datab/msg.vocab \
--validation-source="datab/diff.valid" \
--validation-target="datab/msg.valid" \
--test-source="datab/diff.test" \
--test-target="datab/msg.test" \
--max-source-sentence=500 \
--validation-interval=200 \
--max-target-sentence=300  


python3 seq2seq.py --gpu=1 --epoch=80 --unit=1024 --layer=3 \
-o=" all_bdata/seq/2" --seed=2 --l2=0.000001 \
datab/diff.train \
datab/msg.train \
datab/diff.vocab \
datab/msg.vocab \
--validation-source="datab/diff.valid" \
--validation-target="datab/msg.valid" \
--test-source="datab/diff.test" \
--test-target="datab/msg.test" \
--max-source-sentence=500 \
--validation-interval=200 \
--max-target-sentence=300  


python3 seq2seq.py --gpu=1 --epoch=80 --unit=1024 --layer=3 \
-o=" all_bdata/seq/3" --seed=3 --l2=0.000001 \
datab/diff.train \
datab/msg.train \
datab/diff.vocab \
datab/msg.vocab \
--validation-source="datab/diff.valid" \
--validation-target="datab/msg.valid" \
--test-source="datab/diff.test" \
--test-target="datab/msg.test" \
--max-source-sentence=500 \
--validation-interval=200 \
--max-target-sentence=300  


python3 seq2seq.py --gpu=1 --epoch=80 --unit=1024 --layer=3 \
-o=" all_bdata/seq/4" --seed=4 --l2=0.000001 \
datab/diff.train \
datab/msg.train \
datab/diff.vocab \
datab/msg.vocab \
--validation-source="datab/diff.valid" \
--validation-target="datab/msg.valid" \
--test-source="datab/diff.test" \
--test-target="datab/msg.test" \
--max-source-sentence=500 \
--validation-interval=200 \
--max-target-sentence=300  


python3 seq2seq.py --gpu=1 --epoch=80 --unit=256 --layer=3 \
-o=" all_bdata/join/0" --seed=0 --l2=0.000001 \
datab/di_is.train \
datab/msg.train \
datab/di_is.vocab \
datab/msg.vocab \
--validation-source="datab/di_is.valid" \
--validation-target="datab/msg.valid" \
--test-source="datab/di_is.test" \
--test-target="datab/msg.test" \
--max-source-sentence=500 \
--validation-interval=200 \
--max-target-sentence=300  


python3 seq2seq.py --gpu=1 --epoch=80 --unit=256 --layer=3 \
-o=" all_bdata/join/1" --seed=1 --l2=0.000001 \
datab/di_is.train \
datab/msg.train \
datab/di_is.vocab \
datab/msg.vocab \
--validation-source="datab/di_is.valid" \
--validation-target="datab/msg.valid" \
--test-source="datab/di_is.test" \
--test-target="datab/msg.test" \
--max-source-sentence=500 \
--validation-interval=200 \
--max-target-sentence=300  


python3 seq2seq.py --gpu=1 --epoch=80 --unit=256 --layer=3 \
-o=" all_bdata/join/2" --seed=2 --l2=0.000001 \
datab/di_is.train \
datab/msg.train \
datab/di_is.vocab \
datab/msg.vocab \
--validation-source="datab/di_is.valid" \
--validation-target="datab/msg.valid" \
--test-source="datab/di_is.test" \
--test-target="datab/msg.test" \
--max-source-sentence=500 \
--validation-interval=200 \
--max-target-sentence=300  


python3 seq2seq.py --gpu=1 --epoch=80 --unit=256 --layer=3 \
-o=" all_bdata/join/3" --seed=3 --l2=0.000001 \
datab/di_is.train \
datab/msg.train \
datab/di_is.vocab \
datab/msg.vocab \
--validation-source="datab/di_is.valid" \
--validation-target="datab/msg.valid" \
--test-source="datab/di_is.test" \
--test-target="datab/msg.test" \
--max-source-sentence=500 \
--validation-interval=200 \
--max-target-sentence=300  


python3 seq2seq.py --gpu=1 --epoch=80 --unit=256 --layer=3 \
-o=" all_bdata/join/4" --seed=4 --l2=0.000001 \
datab/di_is.train \
datab/msg.train \
datab/di_is.vocab \
datab/msg.vocab \
--validation-source="datab/di_is.valid" \
--validation-target="datab/msg.valid" \
--test-source="datab/di_is.test" \
--test-target="datab/msg.test" \
--max-source-sentence=500 \
--validation-interval=200 \
--max-target-sentence=300  

