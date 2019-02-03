#! /usr/bin/python3
start_sh =    """
python3 seq2seq.py --gpu=0 --epoch=80 --unit=%d --layer=%d \\
-o="sdata/%d" --seed=%d --l2=0.000001 \\
data5/diff.train \\
data5/msg.train \\
data5/diff.vocab \\
data5/msg.vocab \\
--validation-source="data5/diff.valid" \\
--validation-target="data5/msg.valid" \\
--test-source="data5/diff.test" \\
--test-target="data5/msg.test" \\
--max-source-sentence=500 \\
--validation-interval=200 \\
--max-target-sentence=300  
"""

param = [
[256,3]#,
#[512,3]#,
#[512,2],
#[1024,2],
#[1024,3]
]

data_itr = 30
for i in param:
	for j in range(5):
		print(start_sh % (i[0],i[1],data_itr,j) )
		data_itr += 1
		
