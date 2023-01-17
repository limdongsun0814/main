import tensorflow as tf
import numpy as np
import math
import csv
f = open('distance.csv','r')
rdr = csv.reader(f)
x,y=[],[]
old_y=str(10)
cnt=1
sum_error=0
avg_list=[]
for line in rdr:
	if old_y==str(line[0]):
		cnt=cnt+1
		sum_error=sum_error+float(line[1])
	else:
		print(abs(sum_error/cnt))
		cnt=0
		sum_error=0
		old_y=str(line[0])
		cnt=cnt+1
		sum_error=sum_error+float(line[1])
	#print(line)

print(abs(sum_error/cnt))



