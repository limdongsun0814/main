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
train_set, test_set = train_test_split(xy_data,test_size=0.3)
data_cnt=360
# print(train_set[1])
x_train_data =train_set.T[:data_cnt]
x_train_data_norm = preprocessing.normalize(x_train_data, norm='l2')
y_train_data =train_set.T[360]

x_test_data = test_set.T[:data_cnt]
x_test_data_norm = preprocessing.normalize(x_test_data, norm='l2')
y_test_data = test_set.T[360]
print("[x_train_data]",x_train_data.shape)
print("[y_train_data]", y_train_data.shape)
print("[x_test_data]", x_test_data.shape)
print("[y_test_data]", y_test_data.shape)
# print(x_train_data)
#modeling
model = models.Sequential()

model.add(layers.Dense(data_cnt, input_shape=(data_cnt,),activation="relu"))
# model.add(Dropout(0.9))
model.add(Dropout(0.5))
model.add(layers.Dense(180,activation="relu"))
model.add(Dropout(0.5))
# model.add(layers.Dense(64,activation="relu"))
# model.add(Dropout(0.2))
model.add(layers.Dense(90,activation="relu"))
# model.add(Dropout(0.2))
# model.add(layers.Dense(45,activation="relu"))
model.add(Dropout(0.5))
model.add(layers.Dense(2,activation="softmax"))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),#from_logits=True),
              metrics=['accuracy'])
hist=model.fit(x_train_data.T,y_train_data.T, epochs=1000,batch_size=2000,
                    validation_data = (x_test_data.T, y_test_data.T))
print(model.summary())
model.save('modelv3.h5')
performance_train = model.evaluate(x_train_data.T, y_train_data.T)
print('train_Accuracy : ',format(performance_train[1]))
performance_test = model.evaluate(x_test_data.T, y_test_data.T)
print('test_Accuracy : ',format(performance_test[1]))
#print('Test Loss and Accuracy ->', performace_test)

print(hist.history)
fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper right')

acc_ax.plot(hist.history['acc'], 'b', label='train acc')
acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')
acc_ax.set_ylabel('accuracy')
acc_ax.legend(loc='upper left')

plot_model(model, to_file='model.png')
plot_model(model, to_file='model_shapes.png', show_shapes=True)
y=model.predict_classes(x_test_data.T)
cf = confusion_matrix(y_test_data.T, y)
print(cf)
y=model.predict_classes(x_train_data.T)
cf = confusion_matrix(y_train_data.T, y)
print(cf)
print(xy_data.shape)
# model.save_weights('model1.h5')
# reconstructed_model = load_model("model.h5",compile=False)
# print( model.predict_classes(x_train_data_norm.T),
#        reconstructed_model.predict_classes(x_train_data_norm.T))
plt.show()
