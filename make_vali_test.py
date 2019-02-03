#!/usr/bin/python3
import sys
import random

def count_lines(path):
    with open(path) as f:
        return sum([1 for _ in f])
file_names = ["data5_pp/issu","data5_pp/diff","data5_pp/msg"]

i0 = open(file_names[0]+".rm")
i1 = open(file_names[1]+".rm")
i2 = open(file_names[2]+".rm")

valid0 = open(file_names[0]+".valid","w")
valid1 = open(file_names[1]+".valid","w")
valid2 = open(file_names[2]+".valid","w")

test0 = open(file_names[0]+".test","w")
test1 = open(file_names[1]+".test","w")
test2 = open(file_names[2]+".test","w")

train0 = open(file_names[0]+".train","w")
train1 = open(file_names[1]+".train","w")
train2 = open(file_names[2]+".train","w")

valid_n = int(sys.argv[1])
test_n = int(sys.argv[2])

all_num = range(count_lines(file_names[0]+".txt"))

drip = random.sample(all_num,valid_n+test_n)

drip_valid = drip[:valid_n]
drip_test = drip[valid_n:]

count = 0
for i,j,k in zip(i0,i1,i2):
	if count in drip_valid:
		valid0.write(i)
		valid1.write(j)
		valid2.write(k)
	elif count in drip_test:
		test0.write(i)
		test1.write(j)
		test2.write(k)
	else:
		train0.write(i)
		train1.write(j)
		train2.write(k)
	count +=1
		
