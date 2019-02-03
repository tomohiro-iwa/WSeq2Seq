import re

i0 = open("data5_pp/diff.txt")
i1 = open("data5_pp/issu.txt")
i2 = open("data5_pp/msg.txt")

o0 = open("data5_pp/diff.rm","w")
o1 = open("data5_pp/issu.rm","w")
o2 = open("data5_pp/msg.rm","w")

re.compile

count = 0
for i,j,k in zip(i0,i1,i2):
	if len(re.findall(r"merge",k,re.IGNORECASE)) > 0:
		count += 1
		continue
	o0.write(i)
	o1.write(j)
	o2.write(k)
print(count)
