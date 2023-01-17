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

datasets = np.loadtxt('re_data.csv', delimiter=',',skiprows=1)
data = np.loadtxt('data_linear_test2.csv', delimiter=',',skiprows=1)
xy_data = datasets
xy_data_dnn = data

#+35,+70
x_data_dnn =data[:,-5]-40
y_data_dnn =data[:,-6]+35

x_re_data = xy_data[:,0]
y_re_data = xy_data[:,1]


x_xy_data = xy_data[:,2]
y_xy_data = xy_data[:,3]


x_rth_data = xy_data[:,4]
y_rth_data = xy_data[:,5]



plt.scatter(x_re_data,y_re_data, s=5**2, c='r',label='Actual Data')
# plt.scatter(x_xy_data,y_xy_data, s=5**2, c='b',label='x, y prediction')
# plt.scatter(x_rth_data,y_rth_data, s=5**2, c='b',label='r, theta prediction')
plt.scatter(x_data_dnn,y_data_dnn, s=5**2, c='g',label='train,test data')

print(np.average(np.sqrt((x_re_data-x_xy_data)**2+(y_re_data-y_xy_data)**2)))
print(np.average(np.sqrt((x_re_data-x_rth_data)**2+(y_re_data-y_rth_data)**2)))


# plt.plot([x_re_data,x_xy_data],[y_re_data,y_xy_data],c="k")#,linewidth='10')
# plt.plot([x_re_data,x_rth_data],[y_re_data,y_rth_data],c="k")#,linewidth='10')
plt.legend(loc='upper right')
plt.ylim(50,140)
# plt.grid(True)
plt.show()



