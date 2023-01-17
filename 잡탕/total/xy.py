import cv2
import numpy as np
import matplotlib.pylab as plt
import os


datasets = np.loadtxt('x0,y80.csv', delimiter=',',skiprows=1)
# x=datasets[:,600:1000]
x=datasets[:,:1000]
y=datasets[:,-2]
cnt=0
old_y=0
data = []
for i in range(len(x)):
    if old_y != y[i]:#

        data.append(x[i])
        # plt.plot(x[i],c=[cnt,cnt,0],label=int(cnt*40+50))
        # cnt=cnt+0.25
        old_y=y[i]
# plt.legend().
x_arr=np.array(data)
plt.pcolor(x_arr)
# plt.xlabel('Y-axis')
# plt.ylabel('conut')
plt.show()
cv2.waitKey(0)