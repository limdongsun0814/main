import cv2
import numpy as np
import matplotlib.pylab as plt
import os
root_dir = '/home/d109master/robot/aother'
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
for i in img_list:
    # print(i)
    red_hsv = cv2.cvtColor(i, cv2.COLOR_BGR2HSV)
    # img = cv2.imread('box.jpg')
    channels = cv2.split(red_hsv)
    colors = ('b', 'g', 'r')
    for (ch, color) in zip (channels, colors):
        if color == 'b':
            hist = cv2.calcHist([ch], [0], None, [360], [0, 360])
            plt.figure('img_path_list[cnt]')
            plt.plot(hist, color = color)
# cv2.imshow(str(img_path_list[cnt]),ch)
# if cnt == 3:
# break

cnt=cnt+1

root_dir = '/home/d109master/robot/my_robot'
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
for i in img_list:
    # print(i)
    red_hsv = cv2.cvtColor(i, cv2.COLOR_BGR2HSV)
    # img = cv2.imread('box.jpg')
    channels = cv2.split(red_hsv)
    colors = ('b', 'g', 'r')
    for (ch, color) in zip (channels, colors):
        if color == 'b':
            hist = cv2.calcHist([ch], [0], None, [360], [0, 360])
            plt.figure('img_pa1th_list[cnt]')
            plt.plot(hist, color = 'r')
# cv2.imshow(str(img_path_list[cnt]),ch)
# if cnt == 3:
# break

cnt=cnt+1
plt.show()