from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import metrics
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import datasets
# from keras.utils import np_utils
from tensorflow.keras.layers import Dropout
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import os
import cv2
import pandas as pd
import csv
import math
f = open('data_linear_test140_down.csv','w')
wr= csv.writer(f)
robot_class = '/home/d109master/total/my_robot'
img_path_list = []
my_robot_img=[]
aother_img=[]
possible_img_extension = ['.jpg', '.jpeg', '.JPG', '.bmp', '.png']
my_robot_hist_data = []
aother_hist_data = []
class_name = []

def noisy(noise_typ,image):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.9
        amount = 0.034
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)
        noisy = image + image * gauss
        return noisy



def height_to_distance( plx, h):
    # dis = (8406.536 / h) - 4.7694054
    dis = 12.9 / (
                math.tan(math.atan(math.tan(math.radians(41.41)) * h / 480) - math.atan(math.tan(math.radians(41.41)) * \
                                                                                        (186 - plx) / 480)) + math.tan(
            math.atan(math.tan(math.radians(41.41)) * (186 - plx) / 480)))
    return dis


def center_to_angle( ctr):
    # dis = (8406.536 / h) - 4.7694054
    angle = 90 - math.degrees(math.atan(math.tan(math.radians(53.5 / 2)) * ctr / 320))
    return angle
def x_y(distance,angle):
    x =  distance * math.sin(math.radians(angle-90))
    y =  distance * math.cos(math.radians(angle-90))
    return x,y


for (root, dirs, files) in os.walk(robot_class):
    if len(files) > 0:
        for file_name in files:
            if os.path.splitext(file_name)[1] in possible_img_extension:
                img_path = root + '/' + file_name
                # img_path = file_name

                # img_path = img_path.replace('\\', '/')
                img = cv2.imread(img_path)
                # img1=cv2.resize(img,(300,200),interpolation=cv2.INTER_LINEAR)
                my_robot_img.append(img)
                print(img_path)
                img_path_list.append(img_path)
#
for i in range(len(my_robot_img)):
    # print(i)

    img_resize = cv2.resize(my_robot_img[i], (600, 400), interpolation=cv2.INTER_LINEAR)
    hsv_img = cv2.cvtColor(img_resize, cv2.COLOR_BGR2HSV)
    # img = cv2.imread('box.jpg')

    lower_orange = (10, 101, 74)  #
    upper_orange = (77, 225, 166)  #
    thresh = cv2.inRange(hsv_img, lower_orange, upper_orange)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 7))
    thresh_open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    test = np.count_nonzero(thresh_open > 1.0, axis=0)
    test2 = np.count_nonzero(thresh_open > 1.0, axis=1)


    data = test.tolist()
    data = data + test2.tolist()

    cnt, _, stats, centroids = cv2.connectedComponentsWithStats(thresh_open)
    # print(cnt, stats, centroids)
    a=0
    area_max=0
    x_max=0
    y_max=0
    w_max=0
    h_max=0
    x_ctr_max=0
    y_ctr_max=0
    for z in range(1, cnt):
        (x, y, w, h, area) = stats[z]
        x_ctr,y_ctr=centroids[z]
        a=a+1

        if area < 20:
            continue
        if area_max < area:
            area_max = area

            x_max = x
            y_max = y
            w_max = w
            h_max = h
            x_ctr_max,y_ctr_max=x_ctr, y_ctr
    cv2.rectangle(img_resize, (x_max, y_max), (x_max + w_max, y_max + h_max), (0, 255, 255))
    cv2.putText(img_resize, str(a), (x_max, y_max), cv2.FONT_HERSHEY_PLAIN,
                3, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(img_resize,(int(x_ctr_max),int(y_ctr_max)),1,(255,255,255))
    cv2.imshow(img_path_list[i], img_resize)
    print(x_ctr_max)
    dis=height_to_distance(y_max,h_max)
    agl=center_to_angle(320-x_ctr_max)
    if dis <= 140:
        x_r,y_r=x_y(dis,agl)
        data.append(dis)
        data.append(agl)
        data.append(x_max)
        data.append(y_max)
        data.append(w_max)
        data.append(h_max)
        print(y_r,x_r,'d: ',dis,agl-90,h_max,y_max)
        cv2.waitKey(1)
        cv2.destroyAllWindows()


        wr.writerow(data)
#



for i in range(len(my_robot_img)):
    # print(i)

    img_resize = cv2.resize(noisy("s&p", my_robot_img[i]), (600, 400), interpolation=cv2.INTER_LINEAR)
    hsv_img = cv2.cvtColor(img_resize, cv2.COLOR_BGR2HSV)
    # img = cv2.imread('box.jpg')

    lower_orange = (10, 101, 74)  #
    upper_orange = (77, 225, 166)  #
    thresh = cv2.inRange(hsv_img, lower_orange, upper_orange)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 7))
    thresh_open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    test = np.count_nonzero(thresh_open > 1.0, axis=0)
    test2 = np.count_nonzero(thresh_open > 1.0, axis=1)


    data = test.tolist()
    data = data + test2.tolist()

    cnt, _, stats, centroids = cv2.connectedComponentsWithStats(thresh_open)
    # print(cnt, stats, centroids)
    a=0
    area_max=0
    x_max=0
    y_max=0
    w_max=0
    h_max=0
    x_ctr_max=0
    y_ctr_max=0
    for z in range(1, cnt):
        (x, y, w, h, area) = stats[z]
        x_ctr,y_ctr=centroids[z]
        a=a+1

        if area < 20:
            continue
        if area_max < area:
            area_max = area

            x_max = x
            y_max = y
            w_max = w
            h_max = h
            x_ctr_max,y_ctr_max=x_ctr, y_ctr
    cv2.rectangle(img_resize, (x_max, y_max), (x_max + w_max, y_max + h_max), (0, 255, 255))
    cv2.putText(img_resize, str(a), (x_max, y_max), cv2.FONT_HERSHEY_PLAIN,
                3, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(img_resize,(int(x_ctr_max),int(y_ctr_max)),1,(255,255,255))
    cv2.imshow(img_path_list[i], img_resize)
    print(x_ctr_max)
    dis=height_to_distance(y_max,h_max)
    agl=center_to_angle(320-x_ctr_max)

    if dis <= 140:
        x_r,y_r=x_y(dis,agl)
        data.append(dis)
        data.append(agl)
        data.append(x_max)
        data.append(y_max)
        data.append(w_max)
        data.append(h_max)
        print(y_r,x_r,'d: ',dis,agl-90,h_max,y_max)
        cv2.waitKey(1)
        cv2.destroyAllWindows()


        wr.writerow(data)
#
