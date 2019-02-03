
python3 seq2seq.py --gpu=1 --epoch=80 --unit=256 --layer=3 \
-o="sdata/0" --seed=0 --l2=0.000001 \
data5/diff.train \
data5/msg.train \
data5/diff.vocab \
data5/msg.vocab \
--validation-source="data5/diff.valid" \
--validation-target="data5/msg.valid" \
--test-source="data5/diff.test" \
--test-target="data5/msg.test" \
--max-source-sentence=300 \
--validation-interval=200 \
--max-target-sentence=200  


python3 seq2seq.py --gpu=1 --epoch=80 --unit=256 --layer=3 \
-o="sdata/1" --seed=1 --l2=0.000001 \
data5/diff.train \
data5/msg.train \
data5/diff.vocab \
data5/msg.vocab \
--validation-source="data5/diff.valid" \
--validation-target="data5/msg.valid" \
--test-source="data5/diff.test" \
--test-target="data5/msg.test" \
--max-source-sentence=300 \
--validation-interval=200 \
--max-target-sentence=200  


python3 seq2seq.py --gpu=1 --epoch=80 --unit=256 --layer=3 \
-o="sdata/2" --seed=2 --l2=0.000001 \
data5/diff.train \
data5/msg.train \
data5/diff.vocab \
data5/msg.vocab \
--validation-source="data5/diff.valid" \
--validation-target="data5/msg.valid" \
--test-source="data5/diff.test" \
--test-target="data5/msg.test" \
--max-source-sentence=300 \
--validation-interval=200 \
--max-target-sentence=200  


python3 seq2seq.py --gpu=1 --epoch=80 --unit=256 --layer=3 \
-o="sdata/3" --seed=3 --l2=0.000001 \
data5/diff.train \
data5/msg.train \
data5/diff.vocab \
data5/msg.vocab \
--validation-source="data5/diff.valid" \
--validation-target="data5/msg.valid" \
--test-source="data5/diff.test" \
--test-target="data5/msg.test" \
--max-source-sentence=300 \
--validation-interval=200 \
--max-target-sentence=200  


python3 seq2seq.py --gpu=1 --epoch=80 --unit=256 --layer=3 \
-o="sdata/4" --seed=4 --l2=0.000001 \
data5/diff.train \
data5/msg.train \
data5/diff.vocab \
data5/msg.vocab \
--validation-source="data5/diff.valid" \
--validation-target="data5/msg.valid" \
--test-source="data5/diff.test" \
--test-target="data5/msg.test" \
--max-source-sentence=300 \
--validation-interval=200 \
--max-target-sentence=200  


python3 seq2seq.py --gpu=1 --epoch=80 --unit=512 --layer=3 \
-o="sdata/5" --seed=0 --l2=0.000001 \
data5/diff.train \
data5/msg.train \
data5/diff.vocab \
data5/msg.vocab \
--validation-source="data5/diff.valid" \
--validation-target="data5/msg.valid" \
--test-source="data5/diff.test" \
--test-target="data5/msg.test" \
--max-source-sentence=300 \
--validation-interval=200 \
--max-target-sentence=200  


python3 seq2seq.py --gpu=1 --epoch=80 --unit=512 --layer=3 \
-o="sdata/6" --seed=1 --l2=0.000001 \
data5/diff.train \
data5/msg.train \
data5/diff.vocab \
data5/msg.vocab \
--validation-source="data5/diff.valid" \
--validation-target="data5/msg.valid" \
--test-source="data5/diff.test" \
--test-target="data5/msg.test" \
--max-source-sentence=300 \
--validation-interval=200 \
--max-target-sentence=200  


python3 seq2seq.py --gpu=1 --epoch=80 --unit=512 --layer=3 \
-o="sdata/7" --seed=2 --l2=0.000001 \
data5/diff.train \
data5/msg.train \
data5/diff.vocab \
data5/msg.vocab \
--validation-source="data5/diff.valid" \
--validation-target="data5/msg.valid" \
--test-source="data5/diff.test" \
--test-target="data5/msg.test" \
--max-source-sentence=300 \
--validation-interval=200 \
--max-target-sentence=200  


python3 seq2seq.py --gpu=1 --epoch=80 --unit=512 --layer=3 \
-o="sdata/8" --seed=3 --l2=0.000001 \
data5/diff.train \
data5/msg.train \
data5/diff.vocab \
data5/msg.vocab \
--validation-source="data5/diff.valid" \
--validation-target="data5/msg.valid" \
--test-source="data5/diff.test" \
--test-target="data5/msg.test" \
--max-source-sentence=300 \
--validation-interval=200 \
--max-target-sentence=200  


python3 seq2seq.py --gpu=1 --epoch=80 --unit=512 --layer=3 \
-o="sdata/9" --seed=4 --l2=0.000001 \
data5/diff.train \
data5/msg.train \
data5/diff.vocab \
data5/msg.vocab \
--validation-source="data5/diff.valid" \
--validation-target="data5/msg.valid" \
--test-source="data5/diff.test" \
--test-target="data5/msg.test" \
--max-source-sentence=300 \
--validation-interval=200 \
--max-target-sentence=200  


python3 seq2seq.py --gpu=1 --epoch=80 --unit=512 --layer=2 \
-o="sdata/10" --seed=0 --l2=0.000001 \
data5/diff.train \
data5/msg.train \
data5/diff.vocab \
data5/msg.vocab \
--validation-source="data5/diff.valid" \
--validation-target="data5/msg.valid" \
--test-source="data5/diff.test" \
--test-target="data5/msg.test" \
--max-source-sentence=300 \
--validation-interval=200 \
--max-target-sentence=200  


python3 seq2seq.py --gpu=1 --epoch=80 --unit=512 --layer=2 \
-o="sdata/11" --seed=1 --l2=0.000001 \
data5/diff.train \
data5/msg.train \
data5/diff.vocab \
data5/msg.vocab \
--validation-source="data5/diff.valid" \
--validation-target="data5/msg.valid" \
--test-source="data5/diff.test" \
--test-target="data5/msg.test" \
--max-source-sentence=300 \
--validation-interval=200 \
--max-target-sentence=200  


python3 seq2seq.py --gpu=1 --epoch=80 --unit=512 --layer=2 \
-o="sdata/12" --seed=2 --l2=0.000001 \
data5/diff.train \
data5/msg.train \
data5/diff.vocab \
data5/msg.vocab \
--validation-source="data5/diff.valid" \
--validation-target="data5/msg.valid" \
--test-source="data5/diff.test" \
--test-target="data5/msg.test" \
--max-source-sentence=300 \
--validation-interval=200 \
--max-target-sentence=200  


python3 seq2seq.py --gpu=1 --epoch=80 --unit=512 --layer=2 \
-o="sdata/13" --seed=3 --l2=0.000001 \
data5/diff.train \
data5/msg.train \
data5/diff.vocab \
data5/msg.vocab \
--validation-source="data5/diff.valid" \
--validation-target="data5/msg.valid" \
--test-source="data5/diff.test" \
--test-target="data5/msg.test" \
--max-source-sentence=300 \
--validation-interval=200 \
--max-target-sentence=200  


python3 seq2seq.py --gpu=1 --epoch=80 --unit=512 --layer=2 \
-o="sdata/14" --seed=4 --l2=0.000001 \
data5/diff.train \
data5/msg.train \
data5/diff.vocab \
data5/msg.vocab \
--validation-source="data5/diff.valid" \
--validation-target="data5/msg.valid" \
--test-source="data5/diff.test" \
--test-target="data5/msg.test" \
--max-source-sentence=300 \
--validation-interval=200 \
--max-target-sentence=200  


python3 seq2seq.py --gpu=1 --epoch=80 --unit=1024 --layer=2 \
-o="sdata/15" --seed=0 --l2=0.000001 \
data5/diff.train \
data5/msg.train \
data5/diff.vocab \
data5/msg.vocab \
--validation-source="data5/diff.valid" \
--validation-target="data5/msg.valid" \
--test-source="data5/diff.test" \
--test-target="data5/msg.test" \
--max-source-sentence=300 \
--validation-interval=200 \
--max-target-sentence=200  


python3 seq2seq.py --gpu=1 --epoch=80 --unit=1024 --layer=2 \
-o="sdata/16" --seed=1 --l2=0.000001 \
data5/diff.train \
data5/msg.train \
data5/diff.vocab \
data5/msg.vocab \
--validation-source="data5/diff.valid" \
--validation-target="data5/msg.valid" \
--test-source="data5/diff.test" \
--test-target="data5/msg.test" \
--max-source-sentence=300 \
--validation-interval=200 \
--max-target-sentence=200  


python3 seq2seq.py --gpu=1 --epoch=80 --unit=1024 --layer=2 \
-o="sdata/17" --seed=2 --l2=0.000001 \
data5/diff.train \
data5/msg.train \
data5/diff.vocab \
data5/msg.vocab \
--validation-source="data5/diff.valid" \
--validation-target="data5/msg.valid" \
--test-source="data5/diff.test" \
--test-target="data5/msg.test" \
--max-source-sentence=300 \
--validation-interval=200 \
--max-target-sentence=200  


python3 seq2seq.py --gpu=1 --epoch=80 --unit=1024 --layer=2 \
-o="sdata/18" --seed=3 --l2=0.000001 \
data5/diff.train \
data5/msg.train \
data5/diff.vocab \
data5/msg.vocab \
--validation-source="data5/diff.valid" \
--validation-target="data5/msg.valid" \
--test-source="data5/diff.test" \
--test-target="data5/msg.test" \
--max-source-sentence=300 \
--validation-interval=200 \
--max-target-sentence=200  


python3 seq2seq.py --gpu=1 --epoch=80 --unit=1024 --layer=2 \
-o="sdata/19" --seed=4 --l2=0.000001 \
data5/diff.train \
data5/msg.train \
data5/diff.vocab \
data5/msg.vocab \
--validation-source="data5/diff.valid" \
--validation-target="data5/msg.valid" \
--test-source="data5/diff.test" \
--test-target="data5/msg.test" \
--max-source-sentence=300 \
--validation-interval=200 \
--max-target-sentence=200  


python3 seq2seq.py --gpu=1 --epoch=80 --unit=1024 --layer=3 \
-o="sdata/20" --seed=0 --l2=0.000001 \
data5/diff.train \
data5/msg.train \
data5/diff.vocab \
data5/msg.vocab \
--validation-source="data5/diff.valid" \
--validation-target="data5/msg.valid" \
--test-source="data5/diff.test" \
--test-target="data5/msg.test" \
--max-source-sentence=300 \
--validation-interval=200 \
--max-target-sentence=200  


python3 seq2seq.py --gpu=1 --epoch=80 --unit=1024 --layer=3 \
-o="sdata/21" --seed=1 --l2=0.000001 \
data5/diff.train \
data5/msg.train \
data5/diff.vocab \
data5/msg.vocab \
--validation-source="data5/diff.valid" \
--validation-target="data5/msg.valid" \
--test-source="data5/diff.test" \
--test-target="data5/msg.test" \
--max-source-sentence=300 \
--validation-interval=200 \
--max-target-sentence=200  


python3 seq2seq.py --gpu=1 --epoch=80 --unit=1024 --layer=3 \
-o="sdata/22" --seed=2 --l2=0.000001 \
data5/diff.train \
data5/msg.train \
data5/diff.vocab \
data5/msg.vocab \
--validation-source="data5/diff.valid" \
--validation-target="data5/msg.valid" \
--test-source="data5/diff.test" \
--test-target="data5/msg.test" \
--max-source-sentence=300 \
--validation-interval=200 \
--max-target-sentence=200  


python3 seq2seq.py --gpu=1 --epoch=80 --unit=1024 --layer=3 \
-o="sdata/23" --seed=3 --l2=0.000001 \
data5/diff.train \
data5/msg.train \
data5/diff.vocab \
data5/msg.vocab \
--validation-source="data5/diff.valid" \
--validation-target="data5/msg.valid" \
--test-source="data5/diff.test" \
--test-target="data5/msg.test" \
--max-source-sentence=300 \
--validation-interval=200 \
--max-target-sentence=200  


python3 seq2seq.py --gpu=1 --epoch=80 --unit=1024 --layer=3 \
-o="sdata/24" --seed=4 --l2=0.000001 \
data5/diff.train \
data5/msg.train \
data5/diff.vocab \
data5/msg.vocab \
--validation-source="data5/diff.valid" \
--validation-target="data5/msg.valid" \
--test-source="data5/diff.test" \
--test-target="data5/msg.test" \
--max-source-sentence=300 \
--validation-interval=200 \
--max-target-sentence=200  

