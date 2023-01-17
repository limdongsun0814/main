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
# from tensorflow.python.keras.models import save_weights
from sklearn import preprocessing
from tensorflow.keras.utils import plot_model
import os
import cv2
import pandas as pd
import csv
import h5py


datasets = np.loadtxt('data.csv', delimiter=',',skiprows=1)
xy_data = datasets
train_set, test_set = train_test_split(xy_data,test_size=0.2)
data_cnt=2
# print(train_set[1])
x_train_data =train_set.T[0]
# x_train_data_norm = preprocessing.normalize(x_train_data, norm='l2')
y_train_data =train_set.T[data_cnt]
# y_train_data = y_train_data.reshape(-1,1)
# y_train_data = preprocessing.normalize(y_train_data, norm='l2')

x_test_data = test_set.T[0]
# x_test_data_norm = preprocessing.normalize(x_test_data, norm='l2')
y_test_data = test_set.T[data_cnt]
normarl=360
# y_test_data = y_test_data.reshape(-1,1)
# y_test_data = preprocessing.normalize(y_test_data, norm='l2')
print("[x_train_data]",x_train_data.shape)
print("[y_train_data]", y_train_data.shape)
print("[x_train_data]",x_train_data.dtype)
print("[y_train_data]", y_train_data.dtype)
print("[x_test_data]", x_test_data.shape)
print("[y_test_data]", y_test_data.shape)
# print(x_train_data)
#modeling
model = models.Sequential()
print(y_test_data[0]/200)
model.add(layers.Dense(2, input_shape=(1,),activation="tanh"))
# model.add(Dropout(0.9))
# model.add(Dropout(0.5))
# model.add(layers.Dense(32))#,activation="relu"))
# model.add(Dropout(0.5))
# model.add(layers.Dense(128))#,activation="relu"))
# # # # model.add(Dropout(0.2))
# model.add(Dropout(0.5))
# model.add(layers.Dense(16,activation="relu"))
# model.add(Dropout(0.5))
# model.add(layers.Dense(8,activation="relu"))
# model.add(Dropout(0.5))
# model.add(layers.Dense(4,activation="relu"))
# model.add(Dropout(0.5))
# model.add(layers.Dense(8,activation="relu"))
# model.add(Dropout(0.8))
# model.add(layers.Dense(4,activation="relu"))
# model.add(Dropout(0.5))
# model.add(layers.Dense(1,activation="relu"))
# model.add(Dropout(0.2))
# model.add(Dropout(0.5))
# model.add(Dropout(0.2))
# # model.add(layers.Dense(45,activation="relu"))
# model.add(Dropout(0.5))
model.add(layers.Dense(1,activation='tanh'))
optimizer=tf.keras.optimizers.RMSprop(
    learning_rate=0.001,
    rho=0.9,
    momentum=0.0,
    epsilon=1e-07,
    centered=True,
    name='RMSprop'
)

model.compile(optimizer="adam",
              loss='mse',#)#,#tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),    /mse
               metrics=['mae'])
hist=model.fit(x_train_data.T,y_train_data.T/normarl, epochs=50000,batch_size=100000,
                    validation_data = (x_test_data.T, y_test_data.T/normarl))
# print(model.summary())
model.save('modelv9.h5')
performance_train = model.evaluate(x_train_data.T, y_train_data.T/normarl)
# print('train_Accuracy : ',format(performance_train[1]))
performance_test = model.evaluate(x_test_data.T, y_test_data.T/normarl)
# print('test_Accuracy : ',format(performance_test[1]))
#print('Test Loss and Accuracy ->', performace_test)

# print(hist.history['val_mean_absolute_error'])
fig, loss_ax = plt.subplots()
# acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper right')

# acc_ax.plot(hist.history['mean_absolute_error'], 'b', label='train mean_squared_error')
# acc_ax.plot(hist.history['val_mean_absolute_error'], 'g', label='val val_mean_squared_error')
# acc_ax.set_ylabel('mse')
# acc_ax.legend(loc='upper left')

plot_model(model, to_file='model.png')
plot_model(model, to_file='model_shapes.png', show_shapes=True)
y=model.predict(x_test_data.T)

# cf = confusion_matrix(y_test_data.T, y)
# print(cf)
# y=model.predict(x_train_data.T)
# cf = confusion_matrix(y_train_data.T, y)
print(y.T*normarl)
print(y_test_data)
print(y.T*normarl-y_test_data)
fig, aa=plt.subplots()

aa.plot(y.T*normarl-y_test_data,'r*')
# print(xy_data.shape)
# model.save_weights('model1.h5')
# reconstructed_model = load_model("model.h5",compile=False)
# print( model.predict_classes(x_train_data_norm.T),
#        reconstructed_model.predict_classes(x_train_data_norm.T))
plt.show()
