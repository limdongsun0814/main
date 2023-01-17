from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import metrics
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow import keras
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
from tensorflow.python.keras.models import load_model
# from tensorflow.python.keras.models import save_weights
from sklearn import preprocessing
from tensorflow.keras.utils import plot_model
import os
import cv2
import pandas as pd
import csv
import h5py
from scipy.stats import norm

robot_class = '/home/d109master/robot/aother'
img_path_list = []
my_robot_img=[]
aother_img=[]
possible_img_extension = ['.jpg', '.jpeg', '.JPG', '.bmp', '.png']
my_robot_hist_data = []
aother_hist_data = []
class_name = []
x=[]
y=[]
for (root, dirs, files) in os.walk(robot_class):
    if len(files) > 0:
        for file_name in files:
            if os.path.splitext(file_name)[1] in possible_img_extension:
                img_path = root + '/' + file_name
                img = cv2.imread(img_path)
                my_robot_img.append(img)
                img_path_list.append(img_path)
for i in my_robot_img:
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
            x.append(data)



y_arr=np.array(y)
x_arr= np.array(x)
print(y_arr.shape)
print(x_arr.shape)
print(x_arr[ :,0].shape)
print(np.max(x_arr[:, 0]))
bin=np.arange(0,np.max(x_arr[:, 0])+10,(np.max(x_arr[:, 0])+10)/20)
hist,bins=np.histogram(x_arr[:, 0],bins=bin)
print(len(hist))
print(bin)
plt.pcolor(x_arr)#,bins=bin)
plt.colorbar()
plt.show()


# mu,std =norm.fit(x_arr[:, 0])
# p = norm.pdf(x,mu,std)
# plt.plot()
# aother_img=[]
# img_path_list_aother=[]
# robot_class = '/home/d109master/robot/aother'
# for (root, dirs, files) in os.walk(robot_class):
#     if len(files) > 0:
#         for file_name in files:
#             if os.path.splitext(file_name)[1] in possible_img_extension:
#                 img_path = root + '/' + file_name
#                 img = cv2.imread(img_path)
#                 aother_img.append(img)
#                 img_path_list_aother.append(img_path)
# for i in aother_img:
#     print(i)
#     resize_img=cv2.resize(i, (30, 30))
#     x.append(resize_img)
#     y.append(1)
#
# y_arr=np.array(y)
# x_arr= np.array(x)
# print(y_arr.shape)
# print(x_arr.shape)


