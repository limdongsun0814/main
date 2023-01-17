# from tensorflow.keras import layers
from tensorflow.python.keras.models import load_model
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
from tensorflow.python.keras.models import load_model
from sklearn import preprocessing
from tensorflow.keras.utils import plot_model
import os
import cv2
import pandas as pd
import csv
import h5py
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
# robot_class = '/home/d109master/robot/my_robot'
# img_path_list = []
# my_robot_img=[]
# aother_img=[]
# possible_img_extension = ['.jpg', '.jpeg', '.JPG', '.bmp', '.png']
# my_robot_hist_data = []
# aother_hist_data = []
# class_name = []
# x=[]
# y=[]
# for (root, dirs, files) in os.walk(robot_class):
#     if len(files) > 0:
#         for file_name in files:
#             if os.path.splitext(file_name)[1] in possible_img_extension:
#                 img_path = root + '/' + file_name
#                 img = cv2.imread(img_path)
#                 my_robot_img.append(img)
#                 img_path_list.append(img_path)
# for i in my_robot_img:
#     print(i)
#     resize_img=cv2.resize(i, (30, 30))
#     x.append(resize_img)
#     y.append(0)
#
# y_arr=np.array(y)
# x_arr= np.array(x)
# print(y_arr.shape)
# print(x_arr.shape)
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
#
# y_arr=np.array(y)
# x_arr= np.array(x)
# print(y_arr.shape)
# print(x_arr.shape)
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

# print(train_set[1])
# x_train_data =train_set.T[:data_cnt]
# x_train_data_norm = preprocessing.normalize(x_train_data, norm='l2')
# y_train_data =train_set.T[360]
#
# x_test_data = test_set.T[:data_cnt]
# x_test_data_norm = preprocessing.normalize(x_test_data, norm='l2')
# y_test_data = test_set.T[360]
# fil = xy_data[:,-1]
# x_train_data, x_test_data, y_train_data,y_test_data = train_test_split(x_arr,y_arr,test_size=0.3)
# reconstructed_model = load_model("cnn.h5",compile=False)
# print(reconstructed_model.summary())
# print("[x_train_data]",x_train_data.T.shape)
# print("[y_train_data]", y_train_data.T.shape)
# print("[x_test_data]", x_test_data_norm.shape)
# print("[y_test_data]", y_test_data.shape)
# y=model.predict(x_train_data_norm.T)

# print("[y_train_data]", y_train_data.astype(np.int64)[0])
# print("[y]", y.dtype)
# y_pre_train=reconstructed_model.predict_classes(x_train_data)
# y_pre_test=reconstructed_model.predict_classes(x_test_data)
# cf = confusion_matrix(y_test_data,y_pre_test)
# print(y_pre_train)
# print(y_pre_test)
# print(cf)
# for i in range(len(y)):
#     print(y_train_data.T[i],y[i])
# print(cf)

plt.rc('font', size=20) # font size
cf =np.array( [[429,2],[7,1806]])
fig, ax = plot_confusion_matrix(conf_mat=cf,

                                figsize=(5, 5),

                                show_absolute=True,

                                show_normed=True,

                                colorbar=True)

ax.set_title('Master Robot Classification Confusion Matrix')

label_name = ['', 'Master Robot', 'Others Master Robot', 'Master Robot', 'Others Master Robot']

ax.xaxis.set_ticklabels(label_name)

ax.yaxis.set_ticklabels(label_name)

plt.show()


