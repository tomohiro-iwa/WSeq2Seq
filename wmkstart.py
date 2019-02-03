#! /usr/bin/python3
start_sh =    """
python3 WSeq2Seq.py --gpu=0 --epoch=80 --unit=%d --layer=%d \\
-o="all_data5_pp/wseq/%d" --save="save" --seed=%d --l2=0.000001 \\
data5_pp/issu.train \\
data5_pp/diff.train \\
data5_pp/msg.train \\
data5_pp/issu.vocab \\
data5_pp/diff.vocab \\
data5_pp/msg.vocab \\
--validation-source0="data5_pp/issu.valid" \\
--validation-source1="data5_pp/diff.valid" \\
--validation-target="data5_pp/msg.valid" \\
--test-source0="data5_pp/issu.test" \\
--test-source1="data5_pp/diff.test" \\
--test-target="data5_pp/msg.test"
"""
param = [
#[256,3],
[512,3]#,
#[512,2]
]

data_itr = 0
for i in param:
	for j in range(5):
		print(start_sh % (i[0],i[1],data_itr,j) )
		data_itr += 1
		
