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

f = open('distance_angle_pid.csv','r')
print('distance_angle_pid.csv')
rdr = csv.reader(f)
x,y=[],[]
a=1#4
b=2#2
bl = [True, False]
error = []
distance = []
distance_pre = []
angle = []
angle_pre = []
# tar dis: 60 agl: 90
# tar dis: 80 agl: 100
sw = 0 # sw == 1 PID sw != 1 time
times = 0
times_arr = []
plt.rc('font', size=20) # font size

if sw != 1:
    plt.xlabel('step')
    plt.ylabel('sec per step[sec]')
    plt.title("sec per step")
cnt = 0
tar_d = []
tar_a = []
for line in rdr:
    if sw == 1:
        distance.append(float(line[1]))
        angle.append(float(line[2])-90)
        if cnt<=100:
            tar_d.append(60)
            tar_a.append(0)
        else:
            tar_d.append(80)
            tar_a.append(10)
        # plt.plot(float(line[1]))#, 'ko')
    else:

        times_arr.append(float(line[3]))
        tar_d.append(0.115463199615)
        # plt.plot(float(line[3]),'ro')#, "g")
        # plt.plot(float(line[1]))#, 'ko')
        # print(float(line[0]),float(line[4]))
        # plt.plot([float(line[0]), float(line[4])],[float(line[1]), float(line[5])],'k--')
        # error.append(math.sqrt((float(line[0])- float(line[4]))**2+(float(line[1])- float(line[5]))**2))
        # plt.legend(loc=(0.0, 0.0))

    cnt+=1
    if cnt == 200:
        break

# plt.title("PID control")
print(tar_d)
if sw ==1:
    step = np.arange(0,200)*0.11444

    fig, ax1 = plt.subplots()
    ax1.plot(step,tar_a,'y--',label='target angle')#, 'ko')
    ax1.plot(step,angle,'r',label='angle between master robots')#, 'ko')
    ax1.set_ylabel('angle[degree]')
    ax1.set_ylim(-25,25)


    plt.legend(loc=(0.645, 1.0))
    ax2 = ax1.twinx()
    ax2.plot(step,tar_d,'b--',label='target distance')#, "g")
    ax2.plot(step,distance,'g',label='distance between master robots')#, "g")
    ax2.set_ylabel('distance[cm]')
    ax2.set_ylim(50,110)

    # ax1.set_xlabel('step [1 step = 0.11444sec]')
    ax1.set_xlabel('time[sec]')
    plt.legend(loc=(0.0, 1.0))
else:
    plt.plot(times_arr, 'r',marker='o')
    plt.plot(tar_d,'y--')#, 'ko')
    print(sum(times_arr)/len(times_arr))
# plt.ylim(40,110)
# if sw == 1:
#     plt.figure(1)
#     plt.plot(distance,distance_pre,'ro',)
#     distance_pre_np=np.array(distance_pre)
#     print('aa',np.sqrt(np.average(distance_pre_np**2)))
#     plt.xlabel('actual measurement distance [cm]')
#     plt.ylabel('RMSE [cm]')
#     plt.title("distance")
#     plt.figure(2)
#     plt.plot(angle,angle_pre,'ro')
#     plt.xlabel('actual measurement angle [degree]')
#     plt.ylabel('RMSE [degree]')
#     plt.title("angle")
#     angle_pre=np.array(angle_pre)
#     print('bb',np.sqrt(np.average(angle_pre**2)))

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

