#!/usr/bin/python3

import sys 
import json
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('RESULT', help='chainer result file')
parser.add_argument('--data',default="main/loss",
			help='source sentence list for validation')
parser.add_argument('--cmd',default="max min avg",
			help='data pickup command')

args = parser.parse_args()
log_file = json.load(open(args.RESULT))
data = list(map(lambda x:x[args.data] , log_file ))

cmds = args.cmd.split(" ")
for cmd in cmds:
	value = 0
	if cmd == "max":
		value = max(data)
	if cmd == "min":
		value = min(data)
	if cmd == "avg":
		value = sum(data)/len(data)
	print(cmd+":\t"+str(value))

