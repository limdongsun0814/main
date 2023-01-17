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

f = open('data.csv','w')
wr= csv.writer(f)
robot_class = '/home/d109master/robot/my_robot'
img_path_list = []
my_robot_img=[]
aother_img=[]
possible_img_extension = ['.jpg', '.jpeg', '.JPG', '.bmp', '.png']
my_robot_hist_data = []
aother_hist_data = []
class_name = []
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

for i in my_robot_img:
    # print(i)
    red_hsv = cv2.cvtColor(i, cv2.COLOR_BGR2HSV)
    # img = cv2.imread('box.jpg')
    channels = cv2.split(red_hsv)
    colors = ('b', 'g', 'r')
    for (ch, color) in zip (channels, colors):
        if color == 'b':
            hist = cv2.calcHist([ch], [0], None, [360], [0, 360])
            my_robot_hist_data.append([hist,0])
            reshape = np.reshape(hist,360)
            data = reshape.tolist()
            print(reshape)
            data.append(0)
            # class_name.append(0)
            # my_robot_hist_data.append(0)


###################################################
aother_class = '/home/d109master/robot/aother'

for (root, dirs, files) in os.walk(aother_class):
    if len(files) > 0:
        for file_name in files:
            if os.path.splitext(file_name)[1] in possible_img_extension:
                img_path = root + '/' + file_name
                # img_path = file_name

                # img_path = img_path.replace('\\', '/')
                img = cv2.imread(img_path)
                # img1=cv2.resize(img,(300,200),interpolation=cv2.INTER_LINEAR)
                aother_img.append(img)
                print(img_path)
                img_path_list.append(img_path)
for i in aother_img:
    # print(i)
    red_hsv = cv2.cvtColor(i, cv2.COLOR_BGR2HSV)
    # img = cv2.imread('box.jpg')
    channels = cv2.split(red_hsv)
    colors = ('b', 'g', 'r')
    for (ch, color) in zip (channels, colors):
        if color == 'b':
            hist = cv2.calcHist([ch], [0], None, [360], [0, 360])
            aother_hist_data.append([hist,1])
            reshape = np.reshape(hist,360)
            data = reshape.tolist()
            print(reshape)
            data.append(1)
            # class_name.append(1)
            # aother_hist_data.append(1)
