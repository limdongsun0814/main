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
tf.enable_eager_execution()

datasets = np.loadtxt('data_linear_test2.csv', delimiter=',',skiprows=1)
xy_data = datasets
# train_set, test_set = train_test_split(xy_data,test_size=0.3)
# data_cnt=360
# print(train_set[1])
# x_train_data =train_set.T[:data_cnt]
# x_train_data_norm = preprocessing.normalize(x_train_data, norm='l2')
# y_train_data =train_set.T[360]
#
# x_test_data = test_set.T[:data_cnt]
# x_test_data_norm = preprocessing.normalize(x_test_data, norm='l2')
# y_test_data = test_set.T[360]
# f = open('error.csv','w')
# wr= csv.writer(f)
x_data = xy_data[:,:-6]
y_data = xy_data[:,-6:-4]
normarl_y=350.0  #400 d
normarl_x=100.0
normarl_y=270.0  #400 d
normarl_x=40.0
# # fil = xy_data[:,-1]
# print(x_data.shape)
# print(y_data.shape)

x_data_norm = preprocessing.normalize(x_data, norm='l1',axis=1)




reconstructed_model = load_model("aa.h5",compile=False)
# print(reconstructed_model.summary())
# print("[x_train_data]",x_train_data.T.shape)
# print("[y_train_data]", y_train_data.T.shape)
# print("[x_test_data]", x_test_data_norm.shape)
# print("[y_test_data]", y_test_data.shape)
# y=model.predict(x_train_data_norm.T)

# print("[y_train_data]", y_train_data.astype(np.int64)[0])
# print("[y]", y.dtype)
y=reconstructed_model.predict(x_data_norm)



y_np = np.array(y)
y_np = np.array(y).reshape(y_np.shape[0], (y_np.shape[1]*y_np.shape[2]))
y_a = y_np.T*(normarl_y,normarl_x)
print(y_a.T.shape)
mse = tf.keras.losses.MeanSquaredError()
print(mse(y_data.T[1], y_a.T[1]).numpy())
print(y_a.T[1].shape)
print(y_a.T[1]-40,y_a.T[0]+15)


plt.scatter(y_data.T[1]-40,y_data.T[0]+15, s=2**2, c='r')
plt.scatter(y_a.T[1]-40,y_a.T[0]+15, s=2**2, c='b')
plt.plot([y_data.T[1]-40.0,y_a.T[1]-40.0],[y_data.T[0]+15.0,y_a.T[0]+15.0],color="green")#,linewidth='10')
# plt.plot( y_data.T[1]-40,y_a.T[1] - y_data.T[1], 'b*')
# plt.plot( y_data.T[0]+15,y_a.T[0] - y_data.T[0], 'ro')#  ,,
plt.grid(True)
plt.show()

# for i in range(len(y)):
#     print(y_train_data.T[i],y[i])
# data_csv=np.abs(y_data-y)
# print(y.T*normarl-500-y_test_data)
# for i in range(len(y)):
#     data=[]
#     data.append(y[i]*1000.0-500-y_data[i])
#     # data.append(fil[i])
#     wr.writerow(data)
#
#     print(y[i]*1000.0-500-y_data[i])
# print(cf)

# cf = np.array([[1487,    0],
#  [   1 ,1379]])
# fig, ax = plot_confusion_matrix(conf_mat=cf,
#
#                                 figsize=(5, 5),
#
#                                 show_absolute=True,
#
#                                 show_normed=True,
#
#                                 colorbar=True)
#
# ax.set_title('Confusion Matrix')
#
# label_name = ['', 'my_robot', 'others', 'my_robot', 'others']
#
# ax.xaxis.set_ticklabels(label_name)
#
# ax.yaxis.set_ticklabels(label_name)
#
# plt.show()


