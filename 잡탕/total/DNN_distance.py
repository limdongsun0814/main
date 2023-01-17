from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import metrics
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import datasets
from tensorflow.python.client import device_lib
# from keras.utils import np_utils
from tensorflow.keras.layers import Dropout,SimpleRNN
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
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
datasets = np.loadtxt('data_linear.csv', delimiter=',',skiprows=1)
xy_data = datasets
train_set, test_set = train_test_split(xy_data,test_size=0.3)
data_cnt=1000
# print(train_set[1])
x_train_data =train_set.T[:-6]
x_train_data_norm = preprocessing.normalize(x_train_data, norm='l2')
y_train_data =train_set.T[-6]
#-6,-5
# y_train_data = y_train_data.reshape(-1,1)
# y_train_data = preprocessing.normalize(y_train_data, norm='l2')

x_test_data = test_set.T[:-6]
x_test_data_norm = preprocessing.normalize(x_test_data, norm='l2')
y_test_data = test_set.T[-6]
normarl=1000.0
# y_test_data = y_test_data.reshape(-1,1)
# y_test_data = preprocessing.normalize(y_test_data, norm='l2')
print(y_train_data)
print(y_train_data.shape)
print("[x_train_data]",x_train_data.shape)
print("[x_train_data]",x_train_data)
print("[y_train_data]", y_train_data.shape)
print("[x_train_data]",x_train_data.dtype)
print("[y_train_data]", y_train_data.dtype)
print("[x_test_data]", x_test_data.shape)
print("[y_test_data]", y_test_data.shape)
# print(x_train_data)
#modeling
print(y_test_data[0]/200)
min_loss = 300.0
i_min=0.0
j_min=0.0
k_min=0.0
z_min=0.0
# for i in range(1,100):
#     # for j in range(1,10):
#     #     for k in range(1,10):
#     #         for z in range(1,10):
#         model = models.Sequential()
#         model.add(layers.Dense(100, input_shape=(data_cnt,),activation="tanh"))
#         # model.add(Dropout(0.9))
#         model.add(Dropout(i/100.0))
#         # model.add(layers.Dense(32))#,activation="relu"))
#         # model.add(Dropout(0.8))
#         # model.add(layers.Dense(128,activation="relu"))
#         # model.add(Dropout(0.5))
#         model.add(layers.Dense(50,activation="relu"))
#         model.add(Dropout(i/100.0))
#         model.add(layers.Dense(25,activation="relu"))
#         model.add(Dropout(i/100.0))
#         model.add(layers.Dense(10,activation="relu"))
#         model.add(Dropout(i/100.0))
#         # model.add(Dropout(0.5))
#         # model.add(Dropout(0.2))
#         # # model.add(layers.Dense(45,activation="relu"))
#         # model.add(Dropout(0.5))
#         model.add(layers.Dense(2,activation='tanh'))
#         # optimizer=tf.keras.optimizers.RMSprop(
#         #     learning_rate=0.00001,
#         #     rho=0.1,
#         #     momentum=0.0,
#         #     epsilon=1e-08,
#         #     centered=True,
#         #     name='RMSprop'
#         # )
#
#         model.compile(optimizer='adam',
#                       loss='mse',#)#,#tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#                        metrics=['mae'])
#         hist=model.fit(x_train_data.T,y_train_data.T/normarl, epochs=1000,batch_size=1000,#batch_size=6000,
#                             validation_data = (x_test_data.T, y_test_data.T/normarl))
#         if min_loss >hist.history['loss'][-1]:
#             i_min = i
#             min_loss=hist.history['loss'][-1]
#         print 'Progress: ',(i+1),' %' #((i+1)*(j+1)*(k+1)*(z+1))*10.0/(10.0*10.0*10.0*10.0),' %'


i_min=0.0
#
# for i in range(1,10):
#     model = models.Sequential()
#     model.add(layers.Dense(1000, input_shape=(data_cnt,), activation="tanh"))
#     # model.add(Dropout(0.9))
#     # model.add(Dropout(i_min/100.0))
#     # model.add(layers.Dense(32))#,activation="relu"))
#     model.add(Dropout(0.2))
#     model.add(layers.Dense(100,activation="tanh"))
#     # model.add(Dropout(0.4))
#     model.add(Dropout(0.2))
#     model.add(layers.Dense(10, activation="relu"))
#     model.add(Dropout(0.2))
#     model.add(layers.Dense(5, activation="relu"))
#     model.add(Dropout(0.2))
#     # model.add(layers.Dense(62, activation="relu"))
#     # model.add(Dropout(0.4))
#     # model.add(layers.Dense(62, activation="relu"))
#     # model.add(layers.Dense(31, activation="relu"))
#     # model.add(layers.Dense(15, activation="relu"))
#     # model.add(layers.Dense(7, activation="relu"))
#     # model.add(layers.Dense(3, activation="relu"))
#     # model.add(Dropout(0.2))
#     # model.add(layers.Dense(31, activation="relu"))
#     # model.add(Dropout(0.2))
#     # model.add(Dropout(0.5))
#     # model.add(Dropout(0.2))
#     # # model.add(layers.Dense(45,activation="relu"))
#     # model.add(Dropout(0.2))
#     model.add(layers.Dense(2, activation='tanh'))
#     # optimizer=tf.keras.optimizers.RMSprop(
#         # learning_rate=10**-i,#0.001,
#         #rho=0.9,
#         # momentum=0.9#,
#         #epsilon=None,#1e-08,
#         #centered=True,
#         #name='RMSprop'
#     # )
#
#     optimizer=tf.keras.optimizers.Adam(
#         lr=10**-i, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False
#     )
#     # optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9, decay=0.0, nesterov=True)
#     model.compile(optimizer=optimizer,
#                   loss='mse',  # )#,#tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#                   metrics=['mae'])
#     # for i in range(100):
#
#     hist = model.fit(x_train_data.T, np.arctan(y_train_data.T)/np.pi, epochs=1000, batch_size=1000,  # batch_size=6000,
#                      validation_data=(x_test_data.T, np.arctan(y_test_data.T)/np.pi))
#     if min_loss >hist.history['loss'][-1]:
#         i_min = i
#         min_loss=hist.history['loss'][-1]
# print('-----------------------',i_min,'-----------------------')

model = models.Sequential()
model.add(layers.Dense(1000, input_shape=(data_cnt,), activation="tanh"))
# model.add(Dropout(0.9))
# model.add(Dropout(i_min/100.0))
# model.add(layers.Dense(32))#,activation="relu"))
model.add(Dropout(0.02))
model.add(layers.Dense(500,activation="relu"))
# model.add(Dropout(0.4))
# model.add(Dropout(0.2))
model.add(layers.Dense(100, activation="relu"))
# model.add(Dropout(0.2))
model.add(layers.Dense(10, activation="relu"))
# model.add(Dropout(0.2))
# model.add(layers.Dense(62, activation="relu"))
# model.add(Dropout(0.4))
# model.add(layers.Dense(62, activation="relu"))
# model.add(layers.Dense(31, activation="relu"))
# model.add(layers.Dense(15, activation="relu"))
# model.add(layers.Dense(7, activation="relu"))
# model.add(layers.Dense(3, activation="relu"))
# model.add(Dropout(0.2))
# model.add(layers.Dense(31, activation="relu"))
# model.add(Dropout(0.2))
# model.add(Dropout(0.5))
# model.add(Dropout(0.2))
# # model.add(layers.Dense(45,activation="relu"))
# model.add(Dropout(0.2))
model.add(layers.Dense(1, activation='sigmoid'))
# optimizer=tf.keras.optimizers.RMSprop(
#     learning_rate=10**-6#,#0.001,
#     # rho=0.9,
#     # momentum=0.9,#,
#     #epsilon=None,#1e-08,
#     #centered=False,
# )

optimizer=tf.keras.optimizers.Adam(
    lr=10 ** -4#, beta_1=0.95, beta_2=0.999#, epsilon=None, decay=0.0, amsgrad=False
)
# optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9, decay=0.0, nesterov=True)
model.compile(optimizer=optimizer,
              loss='mse',  # )#,#tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['mae'])
# for i in range(100):

# hist = model.fit(x_train_data.T, np.arctan(y_train_data.T)/np.pi, epochs=10000, batch_size=1000,  # batch_size=6000,
#                  validation_data=(x_test_data.T, np.arctan(y_test_data.T)/np.pi))
hist = model.fit(x_train_data.T, 1/y_train_data.T, epochs=100000, batch_size=1000,  # batch_size=6000,
                 validation_data=(x_test_data.T, 1/y_test_data.T))
# if min_loss >hist.history['loss'][-1]:
#     i_min = i
#     min_loss=hist.history['loss'][-1]
print('-----------------------',i_min,'-----------------------')

# print(model.summary())
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

y=model.predict(x_test_data.T)

# cf = confusion_matrix(y_test_data.T, y)
# print(cf)
# y=model.predict(x_train_data.T)
# cf = confusion_matrix(y_train_data.T, y)
# print(y.T*normarl)
# print(y_test_data)
# print(y.T*normarl-y_test_data)

fig, aa=plt.subplots()
# aa.plot( np.tan(y.T*np.pi)-y_test_data,'r*')
# y=model.predict(x_train_data.T)
# aa.plot( np.tan( y.T*np.pi)-y_train_data,'b*')

aa.plot( 1/y.T-y_test_data,'r*')
y=model.predict(x_train_data.T)
aa.plot( 1/y.T-y_train_data,'b*')

# test_loss_mse=np.mean(np.square(y.T*normarl-y_test_data))
# test_loss_mae=np.mean(np.abs(y.T*normarl-y_test_data))
# print('test_loss_mse: ',test_loss_mse, 'test_loss_mae: ',test_loss_mae)
# print('i min: ',i_min,'j min: ',j_min,'k min: ',k_min,'z min: ',z_min,'loss: ',hist.history['loss'][-1])
# print(xy_data.shape)
# model.save_weights('model1.h5')
# reconstructed_model = load_model("model.h5",compile=False)
# print( model.predict_classes(x_train_data_norm.T),
#        reconstructed_model.predict_classes(x_train_data_norm.T))
plot_model(model, to_file='xyv2.png')
plot_model(model, to_file='xyv3.png', show_shapes=True)
model.save('xyv2.h5')
plt.show()#block=False)
    # plt.pause(0.0000000000000000000000000000001)

