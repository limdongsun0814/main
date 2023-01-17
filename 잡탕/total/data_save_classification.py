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

# f = open('data_classification2_data_augmentation.csv','w')
# wr= csv.writer(f)
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
        # Salt mode  28201 3134
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
        print(len(coords[0]))
        out[coords] = 1
        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
        out[coords] = 0
        print(len(coords[0]))
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
# for i in my_robot_img:
#     # print(i)
#     red_hsv = cv2.cvtColor(i, cv2.COLOR_BGR2HSV)
#     # img = cv2.imread('box.jpg')
#     channels = cv2.split(red_hsv)
#     colors = ('b', 'g', 'r')
#     for (ch, color) in zip (channels, colors):
#         if color == 'b':
#             hist = cv2.calcHist([ch], [0], None, [360], [0, 360])
#             my_robot_hist_data.append([hist,0])
#             reshape = np.reshape(hist,360)
#             data = reshape.tolist()
#             print(reshape.shape)
#             data.append(0)
#             # wr.writerow(data)
#             # class_name.append(0)
#             # my_robot_hist_data.append(0)
#
#
# for i in range(len(my_robot_img)):
#     # print(i)
#     red_hsv = cv2.cvtColor(noisy("s&p", my_robot_img[i]), cv2.COLOR_BGR2HSV)
#     # img = cv2.imread('box.jpg')
#     channels = cv2.split(red_hsv)
#     colors = ('b', 'g', 'r')
#     for (ch, color) in zip (channels, colors):
#         if color == 'b':
#             hist = cv2.calcHist([ch], [0], None, [360], [0, 360])
#             my_robot_hist_data.append([hist,0])
#             reshape = np.reshape(hist,360)
#             data = reshape.tolist()
#             print(reshape.shape)
#             data.append(0)
#             # wr.writerow(data)




###################################################
aother_class = '/home/d109master/total/others'

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

#
# for i in aother_img:
#     # print(i)
#     red_hsv = cv2.cvtColor(i, cv2.COLOR_BGR2HSV)
#     # img = cv2.imread('box.jpg')
#     channels = cv2.split(red_hsv)
#     colors = ('b', 'g', 'r')
#     for (ch, color) in zip (channels, colors):
#         if color == 'b':
#             hist = cv2.calcHist([ch], [0], None, [360], [0, 360])
#             aother_hist_data.append([hist,1])
#             reshape = np.reshape(hist,360)
#             data = reshape.tolist()
#             print(reshape.shape)
#             data.append(1)
#             # wr.writerow(data)
#             # class_name.append(1)
#             # aother_hist_data.append(1)
#
#
# for i in range(len(aother_img)):
#     # print(i)
#     red_hsv = cv2.cvtColor(noisy("s&p", aother_img[i]), cv2.COLOR_BGR2HSV)
#     # img = cv2.imread('box.jpg')
#     channels = cv2.split(red_hsv)
#     colors = ('b', 'g', 'r')
#     for (ch, color) in zip (channels, colors):
#         if color == 'b':
#             hist = cv2.calcHist([ch], [0], None, [360], [0, 360])
#             my_robot_hist_data.append([hist,0])
#             reshape = np.reshape(hist,360)
#             data = reshape.tolist()
#             print(reshape.shape)
#             data.append(1)
#             # wr.writerow(data)


cv2.imwrite('add robot1 noisy.jpg', noisy("s&p", my_robot_img[2000]))
cv2.imwrite('not  noisy.jpg', my_robot_img[1])
