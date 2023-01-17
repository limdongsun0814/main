# import tensorflow as tf
# import matplotlib.pyplot as plt
# import numpy as np
# import math
# import csv
# f = open('distance new.csv','r')
# rdr = csv.reader(f)
# x,y=[],[]
# old_y=str(50)
# cnt=0
# sum_error=0
# data_list=[]
# avg_list=[]
# max_list=[]
# min_list=[]
# y_list=[]
# a=4#4,2
# for line in rdr:
# 	if old_y==str(line[0]):
# 		cnt=cnt+1
# 		sum_error=sum_error+float(line[a])
# 		data_list.append(float(line[a]))
# 	else:
# 		print(abs(sum_error/cnt))
# 		avg_list.append(sum_error/cnt)
# 		cnt=0
# 		sum_error=0
# 		y_list.append(float(old_y))
# 		old_y=str(line[0])
# 		cnt=cnt+1
# 		sum_error=sum_error+float(line[a])
# 		max_list.append(max(data_list))
# 		min_list.append(min(data_list))
# 		data_list = []
# 		#plt.scatter(old_y, sum_error/cnt,c="r")
# 		#plt.scatter(old_y, max(data_list),c="g")
# 		#plt.scatter(old_y, min(data_list),c="b")
#
# #print(line)
#
# avg_list.append(sum_error/cnt)
# max_list.append(max(data_list))
# min_list.append(min(data_list))
# y_list.append(float(old_y))
# y_list_np=np.array(y_list)
# max_list_np=np.array(max_list)
# min_list_np=np.array(min_list)
# avg_list_np=np.array(avg_list)
# print(np.array(y_list))
# #plt.scatter(old_y, sum_error/cnt,c="r")
# #plt.scatter(old_y, max(data_list),c="g")
# #plt.scatter(old_y, min(data_list),c="b")
# plt.plot(y_list_np,max_list_np,c="r",label='max error')
# plt.gca().set_xlim(left=0)
# plt.plot(y_list_np,min_list_np,c="g",label='min error')
# plt.plot(y_list_np,avg_list_np,c="b",label='avg error')
# plt.legend(loc='lower left')
# plt.xlabel('measuring distance [cm]')
# plt.ylabel('error [cm]')
# print(max_list)
# print(min_list)
# print(avg_list)
# #plt.ylim(-7.0,7.0)
# plt.xlim(45,145)
# plt.grid(True,alpha=1)
# plt.show()
# print(abs(sum_error/cnt))
#
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math
import csv
f = open('distance new.csv','r')
rdr = csv.reader(f)
x,y=[],[]
old_y=str(50)
cnt=0
sum_error=0
sum_error_a=0
data_list=[]
avg_list=[]
avg_list_a=[]
max_list=[]
min_list=[]
y_list=[]
data_arr = []
a=8#4
b=7#2
bl = [True, False]
plt.rc('font', size=20) # font size
for line in rdr:
	if old_y==str(line[0]):
		cnt=cnt+1
		sum_error=sum_error+float(line[a])
		sum_error_a=sum_error_a+float(line[b])
		data_list.append(float(line[a]))
	else:
		print(abs(sum_error/cnt))
		print(abs(sum_error_a/cnt))
		sum_error=sum_error+float(line[a])
		sum_error_a=sum_error_a+float(line[b])
		cnt=cnt+1
		avg_list.append(sum_error/cnt)
		avg_list_a.append(sum_error_a/cnt)
		print(cnt)
		cnt=0
		sum_error,sum_error_a=0,0
		y_list.append(float(old_y))
		old_y=str(line[0])
		# sum_error_a=sum_error_a+float(line[b])
		# sum_error=sum_error+float(line[a])
		max_list.append(max(data_list))
		min_list.append(min(data_list))
		data_arr.append(data_list)
		# data_list = []
		# plt.scatter(old_y, sum_error/cnt,c="r")
		# plt.scatter(old_y, max(data_list),c="g")
		# plt.scatter(old_y, min(data_list),c="b")

#print(line)

avg_list.append(sum_error/cnt)
avg_list_a.append(sum_error_a/cnt)
max_list.append(max(data_list))
min_list.append(min(data_list))
y_list.append(float(old_y))
y_list_np=np.array(y_list)
max_list_np=np.array(max_list)
min_list_np=np.array(min_list)
avg_list_np=np.array(avg_list)
avg_list_np_a=np.array(avg_list_a)
print(np.array(y_list))
#plt.scatter(old_y, sum_error/cnt,c="r")
#plt.scatter(old_y, max(data_list),c="g")
#plt.scatter(old_y, min(data_list),c="b")

plt.gca().set_xlim(left=0)
# for k in bl:
# 	for i in bl:
# 		for j in bl:
# 			for z in bl:
# 				if i== True:
# 					plt.plot(y_list_np,max_list_np,c="r",label='max error')
# 					plt.plot(y_list_np,min_list_np,c="g",label='min error')
# 					plt.legend(loc='upper left')
#
# 				if j == True:
# 					plt.plot(y_list_np,avg_list_np*0,c="k",alpha=1)
# 				if z == True:
# 					arrowprops = dict(color='r',arrowstyle="->")
# 					for ii,jj in zip(avg_list_np,y_list_np):
# 						plt.annotate('',xy=(jj,ii),xytext=(jj,0),arrowprops=arrowprops)
#
#
#
# 				plt.plot(y_list_np,avg_list_np,c="b",label='avg error')
# 				plt.xlabel('measuring angle [degree]')
# 				plt.ylabel('error [degree]')
# 				plt.ylim(-3,3)
# 				plt.xlim(70,110)
# 				plt.grid(False,alpha=1)
# 				name = '0 '+str(False)+' '+str(i)+' ' +str(j)+' ' +str(z) +'.png'
# 				plt.savefig(name)
# 				plt.close()
print(avg_list_np_a)
print(avg_list_np)
# plt.plot(y_list_np,avg_list_np,c="r",label='After Correction',marker='o')
# plt.plot(y_list_np,avg_list_np_a,c="b",label='Before Correction',marker='o')
plt.plot(y_list_np,avg_list_np-(y_list_np),c="r",label='After Correction',marker='o')
plt.plot(y_list_np,avg_list_np_a-(y_list_np),c="b",label='Before Correction',marker='o')

plt.axhline(y=-0.001, color='k', linestyle='-',linewidth=3)
plt.legend(loc=[0,1])
plt.xlabel('measuring distance [cm]')
plt.ylabel('prediction [cm]')
# plt.plot(data_np,y_np)
# plt.plot(data_np_b,y_np_b)
# plt.plot(data_list,y_list,'o')
# plt.plot(data_list_b,y_list_b,'o')
# plt.ylim(-25,5)
plt.xlim(40,145)
# plt.ylim(40,145)
# plt.title("Range of Travel Available")
plt.grid(False,alpha=1)

print(max_list)
print(min_list)
print(avg_list)
#plt.grid(True,alpha=1)
plt.show()
print(abs(sum_error/cnt))

