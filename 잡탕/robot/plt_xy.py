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
sw = 0 # sw == 1 error sw != 1 xy

plt.rc('font', size=15) # font size

if sw != 1:
    plt.xlabel('x-axis [cm]')
    plt.ylabel('y-axis [cm]')
    plt.title("Comparison of measured and predicted coordinates")
    plt.plot(0.0,40.0, "ro",label='Master robot has partially deviated')
    plt.plot(0.0,40.0, "bo",label='Detection of missing objects')
    plt.plot(0.0,40.0, "yo",label='out of range is Master Robot')
    plt.plot(0.0,40.0, "go",label='Real Measured Coordinates')
    plt.plot(0.0,40.0, "ko",label='Predictive Coordinates')

for line in rdr:

    #,label='measuring distance')
    if line[2]=='a':
        if sw != 1:
            pass
            # plt.plot(float(line[0]),float(line[1]),"ro")#,label='ideal measuring distance')
        else:
            pass
    elif line[2]=='b':
        if sw != 1:
            pass
            # plt.plot(float(line[0]),float(line[1]), "bo")#,label='test')
        else:
            pass
    elif line[2]=='c':
        if sw != 1:
            pass
            # plt.plot(float(line[0]),float(line[1]), "yo")
        else:
            pass
    else:
        if sw == 1:
            distance.append(float(line[6]))
            angle.append(float(line[7])-90.0)
            distance_pre.append(math.sqrt((float(line[2])-float(line[6]))**2))
            angle_pre.append(math.sqrt((float(line[3])-float(line[7]))**2))
            # distance_pre.append(((float(line[2])-float(line[6]))))
            # angle_pre.append(((float(line[3])-float(line[7]))))
            print(float(line[2])-float(line[7]))
        else:
            plt.plot(float(line[0]),float(line[1]), "ro")
            plt.plot(float(line[4]),float(line[5]), 'bo')
            print(float(line[0]),float(line[4]))
            plt.plot([float(line[0]), float(line[4])],[float(line[1]), float(line[5])],'k--')
            error.append(math.sqrt((float(line[0])- float(line[4]))**2+(float(line[1])- float(line[5]))**2))
            plt.legend(loc=(0.77, 1.0))
if sw == 1:
    plt.figure(1)
    plt.plot(distance,distance_pre,'ro',)
    distance_pre_np=np.array(distance_pre)
    print('aa',np.average(distance_pre_np))
    plt.xlabel('actual measurement distance [cm]')
    plt.ylabel('RMSE [cm]')
    plt.title("distance")
    plt.figure(2)
    plt.plot(angle,angle_pre,'ro')
    plt.xlabel('actual measurement angle [degree]')
    plt.ylabel('RMSE [degree]')
    plt.title("angle")
    angle_pre=np.array(angle_pre)
    print('bb',np.average(angle_pre))
    print('cc',np.max(angle_pre))
# print(sum(error)/len(error))

# shp=patches.Wedge((0,0),140,70,110, color=[0,1,1])
# plt.gca().add_patch(shp)

plt.show()


## code test##
# plt.legend(loc='upper left')
# plt.plot(data_np,y_np)
# plt.plot(data_np_b,y_np_b)
# plt.plot(data_list,y_list,'o')
# plt.plot(data_list_b,y_list_b,'o')
# plt.ylim(-25,5)
# plt.xlim(40,145)
# plt.ylim(40,145)
# plt.title("Range of Travel Available")
# print(sum(error)/len(error))
# plt.grid(False,alpha=1)

#plt.grid(True,alpha=1)

