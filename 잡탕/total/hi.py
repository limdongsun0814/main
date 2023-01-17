import cv2
import numpy as np
import matplotlib.pylab as plt
import os
root_dir = '/home/d109master/total/others'
# root_dir = '/home/d109master/total/2'
img_path_list = []
img_list=[]
possible_img_extension = ['.jpg', '.jpeg', '.JPG', '.bmp', '.png']

for (root, dirs, files) in os.walk(root_dir):
    if len(files) > 0:
        for file_name in files:
            if os.path.splitext(file_name)[1] in possible_img_extension:
                img_path = root + '/' + file_name
                # img_path = file_name

                # img_path = img_path.replace('\\', '/')
                img = cv2.imread(img_path)
                # img1=cv2.resize(img,(300,200),interpolation=cv2.INTER_LINEAR)
                img_list.append(img)
                print(img_path)
                img_path_list.append(img_path)
cnt=0
a= 4
print(cnt)
hist_list_all=[]

for i in img_list:
    # print(i)
    red_hsv = cv2.cvtColor(i, cv2.COLOR_BGR2HSV)
    # img = cv2.imread('box.jpg')
    channels = cv2.split(red_hsv)
    colors = ('b', 'g', 'r')
    for (ch, color) in zip (channels, colors):
        if color == 'b':
            hist = cv2.calcHist([ch], [0], None, [360], [0, 360])
            hist_list=hist.tolist()
            hist_list_all.append(hist_list)
            if cnt==0:
                hist_frist=hist
            cnt=cnt+1
            # print(hist.shape)
            # plt.figure('img_pa1th_list[cnt]')
            # plt.plot(hist, 'r.')
# cv2.imshow(str(img_path_list[cnt]),ch)
# if cnt == 3:
# break
hist_np= np.array(hist_list_all)
hist_np_2D = hist_np.reshape(hist_np.shape[0], (hist_np.shape[1]*hist_np.shape[2]))
hist_avg=np.average(hist_np_2D,axis=0)
hist_min=np.min(hist_np_2D,axis=0)
hist_max=np.max(hist_np_2D,axis=0)


print(hist_avg.shape)
# plt.figure('avg')
plt.plot(hist_frist, color='b',label='others')
cv2.imshow('hist1',img_list[0])
# plt.figure('min')
# plt.plot(hist_min, color='r')
# plt.figure('max')
# plt.plot(hist_max, color='b')


root_dir = '/home/d109master/total/my_robot'
# root_dir = '/home/d109master/total/1'
img_path_list = []
img_list=[]
possible_img_extension = ['.jpg', '.jpeg', '.JPG', '.bmp', '.png']

for (root, dirs, files) in os.walk(root_dir):
    if len(files) > 0:
        for file_name in files:
            if os.path.splitext(file_name)[1] in possible_img_extension:
                img_path = root + '/' + file_name
                # img_path = file_name

                # img_path = img_path.replace('\\', '/')
                img = cv2.imread(img_path)
                # img1=cv2.resize(img,(300,200),interpolation=cv2.INTER_LINEAR)
                img_list.append(img)
                print(img_path)
                img_path_list.append(img_path)

cnt=0
a= 4
print(cnt)
hist_list_all=[]

for i in img_list:
    # print(i)
    red_hsv = cv2.cvtColor(i, cv2.COLOR_BGR2HSV)
    # img = cv2.imread('box.jpg')
    channels = cv2.split(red_hsv)
    colors = ('b', 'g', 'r')
    for (ch, color) in zip (channels, colors):
        if color == 'b':
            hist = cv2.calcHist([ch], [0], None, [360], [0, 360])
            hist_list=hist.tolist()
            hist_list_all.append(hist_list)
            if cnt==0:
                hist_frist=hist
            cnt=cnt+1
            # print(hist.shape)
            # plt.figure('img_pa1th_list[cnt]')
            # plt.plot(hist, 'r.')
# cv2.imshow(str(img_path_list[cnt]),ch)
# if cnt == 3:
# break
hist_np= np.array(hist_list_all)
hist_np_2D = hist_np.reshape(hist_np.shape[0], (hist_np.shape[1]*hist_np.shape[2]))
hist_avg=np.average(hist_np_2D,axis=0)
hist_min=np.min(hist_np_2D,axis=0)
hist_max=np.max(hist_np_2D,axis=0)


print(hist_avg.shape)
# plt.figure('avg')
plt.plot(hist_frist, color='r',label='my robot')
# plt.figure('min')
# plt.plot(hist_min, color='r')
# plt.figure('max')
# plt.plot(hist_max, color='b')
# plt.xlim(50,70)
cnt=cnt+1
cv2.imshow('hist2',img_list[0])
plt.legend()
plt.show()
cv2.waitKey(0)