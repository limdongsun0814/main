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
import matplotlib.patches as patches

f = open('xy3.csv','r')
print('xy3.csv')
rdr = csv.reader(f)
x,y=[],[]
a=8#4
b=7#2
bl = [True, False]
error = []
distance = []
distance_pre = []
angle = []
angle_pre = []
sw = 1 # sw == 1 error sw != 1 xy

plt.rc('font', size=15) # font size


# 10 ~ 200
plex = np.arange(70,142)
fx = 12.9/(np.tan(np.arctan(np.tan(math.radians(41.41))*plex/480.0)-np.arctan(np.tan(math.radians(41.41)*(186-126)/480))))
x= fx
y= fx/plex
plt.plot(x,y,'r')

plt.xlabel('distance [cm]')
plt.ylabel('distance per pixel [cm]')
plt.show()

