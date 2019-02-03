
i0 = open("datab/diff2")
i1 = open("datab/issu2")
i2 = open("datab/msg2")

o0 = open("datab/diff.txt","w")
o1 = open("datab/issu.txt","w")
o2 = open("datab/msg.txt","w")


null = [0,0,0]
for i,j,k in zip(i0,i1,i2):
	if i=="\n" or j=="\n" or k=="\n":
		if i=="\n":
			null[0]+=1
		if j=="\n":
			null[1]+=1
		if k=="\n":
			null[2]+=1
		continue
	o0.write(i)
	o1.write(j)
	o2.write(k)
print(null)
