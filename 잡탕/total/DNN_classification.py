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

class CustomLearningRateScheduler(keras.callbacks.Callback):
    """Learning rate scheduler which sets the learning rate according to schedule.

  Arguments:
      schedule: a function that takes an epoch index
          (integer, indexed from 0) and current learning rate
          as inputs and returns a new learning rate as output (float).
  """

    def __init__(self, schedule):
        super(CustomLearningRateScheduler, self).__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer.
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(epoch, lr)
        # Set the value back to the optimizer before this epoch starts
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        print("\nEpoch %05d: Learning rate is %6.4f." % (epoch, scheduled_lr))

def lr_schedule(epoch, lr):
    """Helper function to retrieve the scheduled learning rate based on epoch."""
    if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
        return lr
    for i in range(len(LR_SCHEDULE)):
        if epoch == LR_SCHEDULE[i][0]:
            return LR_SCHEDULE[i][1]
    return lr
# datasets = np.loadtxt('data_classification2.csv', delimiter=',',skiprows=1)
datasets = np.loadtxt('data_classification2_data_augmentation.csv', delimiter=',',skiprows=1)
xy_data = datasets
train_set, test_set = train_test_split(xy_data,test_size=0.3)
data_cnt=360
# print(train_set[1])
x_train_data =train_set.T[:data_cnt]
x_train_data_norm = preprocessing.normalize(x_train_data, norm='l2',axis=0)
y_train_data =train_set.T[360]

x_test_data = test_set.T[:data_cnt]
x_test_data_norm = preprocessing.normalize(x_test_data, norm='l2',axis=0)
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
# model.add(Dropout(0.5))
model.add(Dropout(0.5))
model.add(layers.Dense(180,activation="relu"))
model.add(Dropout(0.5))
# model.add(layers.Dense(64,activation="relu"))
# model.add(Dropout(0.5))
model.add(layers.Dense(90,activation="relu"))
model.add(Dropout(0.5))
# model.add(Dropout(0.5))
# model.add(layers.Dense(45,activation="relu"))
# model.add(Dropout(0.2))
model.add(layers.Dense(2,activation="softmax"))

optimizer=tf.keras.optimizers.Adam(
    lr=10 ** -3#, beta_1=0.95, beta_2=0.999#, epsilon=None, decay=0.0, amsgrad=False
)

model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),#from_logits=True),
              metrics=['accuracy'])
LR_SCHEDULE = [
    # (epoch to start, learning rate) tuples
    # (1, 10**-2),
    (1500, 10**-4),
    # (1000, 10**-4),
    # (2000, 10**-7),
]
#
# hist=model.fit(x_train_data.T,y_train_data.T, epochs=2000,batch_size=1000,
#                     validation_data = (x_test_data.T, y_test_data.T))

hist = model.fit(x_train_data.T,y_train_data.T, epochs=3000,batch_size=1000,
                    validation_data = (x_test_data.T, y_test_data.T),
                 callbacks=[
    #LossAndErrorPrintingCallback(),
    CustomLearningRateScheduler(lr_schedule),
])
# model.save('modelsee.h5')
# model.save_weights('modelsee.h5')

# print(model.summary())
# model.save('classification2.h5')
performance_train = model.evaluate(x_train_data.T, y_train_data.T)
# print('train_Accuracy : ',format(performance_train[1]))
performance_test = model.evaluate(x_test_data.T, y_test_data.T)
# print('test_Accuracy : ',format(performance_test[1]))
#print('Test Loss and Accuracy ->', performace_test)

# print(hist.history)
# fig, loss_ax = plt.subplots()
# acc_ax = loss_ax.twinx()
#
# loss_ax.plot(hist.history['loss'], 'y', label='train loss')
# loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
# loss_ax.set_xlabel('epoch')
# loss_ax.set_ylabel('loss')
# loss_ax.legend(loc='upper right')
# fig.ylim(0,0.5)

# fig, loss_ax = plt.subplots()
# acc_ax = loss_ax.twinx()

plt.plot(hist.history['loss'], 'y', label='train loss')
plt.plot(hist.history['val_loss'], 'r', label='val loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='upper right')
plt.ylim(-0.001,2)
# acc_ax.plot(hist.history['acc'], 'b', label='train acc')
# acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')
# acc_ax.set_ylabel('accuracy')
# acc_ax.legend(loc='upper left')

# plot_model(model, to_file='model.png')
# plot_model(model, to_file='model_shapes.png', show_shapes=True)
y=model.predict_classes(x_test_data.T)
cf = confusion_matrix(y_test_data.T, y)
print(cf)
y=model.predict_classes(x_train_data.T)
cf = confusion_matrix(y_train_data.T, y)
print(cf)
# print(xy_data.shape)
# reconstructed_model = load_model("modelsee.h5",compile=False)
# # print( model.predict_classes(x_train_data_norm.T),
# #        reconstructed_model.predict_classes(x_train_data_norm.T))
plt.show()
