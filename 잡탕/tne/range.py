import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math
import csv
f = open('range.csv','r')
rdr = csv.reader(f)
x,y=[],[]
old_y=str(50)
cnt=0
sum_error=0
data_list=[]
y_list=[]
data_list_b=[]
y_list_b=[]
a=4#4,2
min= 300
max = 0
old = -100
list = []
for line in rdr:
    if line[2] =='2':
        if int(line[0]) !=90:
            print(line[0],line[1])
            if cnt == 0:
                data_list.append(int(line[0]))
                y_list.append(int(line[1]))
                cnt=1
            else:
                data_list_b.append(int(line[0]))
                y_list_b.append(int(line[1]))
                cnt=0
        else:
            data_list.append(int(line[0]))
            y_list.append(int(line[1]))



data_np=np.array(data_list)
y_np=np.array(y_list)
data_np_b=np.array(data_list_b)
y_np_b=np.array(y_list_b)

y_np_all= np.append(y_np_b,y_np[::-1])
y_np_all= np.append(y_np_all,y_np_b[0])
data_np_all = np.append(data_np_b,data_np[::-1])
data_np_all = np.append(data_np_all,data_np_b[0])
plt.plot(0,0,'ro', label='Master Robot')
plt.plot(data_np_all,y_np_all,label='Measurement limit')
plt.legend(loc='lower left')
plt.xlabel('Side Displacement [cm]')
plt.ylabel('Linear Displacement [cm]')
# plt.plot(data_np,y_np)
# plt.plot(data_np_b,y_np_b)
# plt.plot(data_list,y_list,'o')
# plt.plot(data_list_b,y_list_b,'o')
plt.ylim(-5,250)
plt.xlim(-100,100)
plt.title("Range of Travel Available")
plt.grid(False,alpha=1)
plt.show()

