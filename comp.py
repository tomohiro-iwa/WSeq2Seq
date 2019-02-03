import sys
for l,r in zip(open(sys.argv[1]), open(sys.argv[2])):
	if float(l) > float(r):
		print(">")
	elif float(l) == float(r):
		print("==")
	elif float(l) < float(r):
		print(float(r)-float(l))
