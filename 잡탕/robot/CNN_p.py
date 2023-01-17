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

img_path_list = []
my_robot_img=[]
aother_img=[]
my_robot_hist_data = []
aother_hist_data = []
class_name = []
x=[]
y=[]
video_name=[]
for i in range(0,341,20):
    name= 'p/'+str(i)+'.avi'
    print(name)
    video_name.append(name)

for i in video_name:
    cap = cv2.VideoCapture(i)
    cnt=3500
    dir= '0'
    x_avg=[]
    x_sum=0
    while cap.isOpened():
        ret, frame = cap.read()
        # print(None==frame)
        # print('type',frame is None)#,type(frame),len(frame))
        # if   all(frame == None)==True:
        #     break
        if (cv2.waitKey(1) & 0xFF == ord('q')) or (frame is None):
            print('stop')
            break


        img_resize = cv2.resize(frame, (30, 30), interpolation=cv2.INTER_LINEAR)
        hsv_img = cv2.cvtColor(img_resize, cv2.COLOR_BGR2HSV)
        # img = cv2.imread('box.jpg')

        lower_orange = (10, 101, 74)  #
        upper_orange = (77, 225, 166)  #
        thresh = cv2.inRange(hsv_img, lower_orange, upper_orange)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 7))
        thresh_open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        test = np.count_nonzero(thresh_open > 1.0, axis=0)
        test2 = np.count_nonzero(thresh_open > 1.0, axis=1)
        data = test.tolist()
        data = data + test2.tolist()
        x.append(data)
        y.append(int(i[2:-4]))
        print(i[2:-4])
        my_robot_img.append(img_resize)
        cv2.imshow('video', frame)

y_arr=np.array(y)
# x_arr= np.array(x)
x_arr= np.array(my_robot_img)
print(y_arr.shape)
print(x_arr.shape)
# xy_data = np.concatenate((x_arr,y_arr),axis=1)
x_train_data, x_test_data, y_train_data,y_test_data = train_test_split(x_arr,y_arr,test_size=0.3)
print(x_train_data.shape)
print(x_test_data.shape)
print(y_train_data.shape)
print(y_test_data.shape)
# data_cnt=360
# # print(train_set[1])
# x_train_data =train_set.T[:-1]
# x_train_data_norm = preprocessing.normalize(x_train_data, norm='l2',axis=0)
# y_train_data =train_set.T[-1]
#
# x_test_data = test_set.T[:-1]
# x_test_data_norm = preprocessing.normalize(x_test_data, norm='l2',axis=0)
# y_test_data = test_set.T[-1]
# print("[x_train_data]",x_train_data.shape)
# print("[y_train_data]", y_train_data.shape)
# print("[x_test_data]", x_test_data.shape)
# print("[y_test_data]", y_test_data.shape)
# # print(x_train_data)
# #modeling
#
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(30, 30, 3)))
# model.add(layers.Dense(500, activation='relu', input_shape=1000))

# model.add(layers.Dense(500, input_shape=(1000,),activation="tanh"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
# model.add(Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
# model.add(Dropout(0.5))
model.add(layers.Dense(32, activation='relu'))
# model.add(Dropout(0.5))
model.add(layers.Dense(16, activation='relu'))
# model.add(Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
#
#
optimizer=tf.keras.optimizers.Adam(
    lr=10 ** -4#, beta_1=0.95, beta_2=0.999#, epsilon=None, decay=0.0, amsgrad=False
)
#
model.compile(optimizer=optimizer,
              loss='mse',#from_logits=True),
              metrics=['mse'])
LR_SCHEDULE = [
    # (epoch to start, learning rate) tuples
    # (1, 10**-2),
    (1500, 10**-3),
    # (1000, 10**-4),
    # (2000, 10**-7),
]
# #
# # hist=model.fit(x_train_data.T,y_train_data.T, epochs=2000,batch_size=1000,
# #                     validation_data = (x_test_data.T, y_test_data.T))
#
plot_model(model, to_file='CNNp.png')
plot_model(model, to_file='CNNp1.png', show_shapes=True)
hist = model.fit(x_train_data,y_train_data/360.0, epochs=1000,batch_size=1000,
                    validation_data = (x_test_data, y_test_data/360.0))
model.save('pdnn.h5')
# # model.save_weights('modelsee.h5')
#
# # print(model.summary())
# # model.save('classification2.h5')
# performance_train = model.evaluate(x_train_data.T, y_train_data.T)
# # print('train_Accuracy : ',format(performance_train[1]))
# performance_test = model.evaluate(x_test_data.T, y_test_data.T)
# # print('test_Accuracy : ',format(performance_test[1]))
# #print('Test Loss and Accuracy ->', performace_test)
#
# # print(hist.history)
fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()
# #
loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper right')
# # fig.ylim(0,0.5)
#
# fig, loss_ax = plt.subplots()
# acc_ax = loss_ax.twinx()
#
# plt.plot(hist.history['loss'], 'y', label='train loss')
# plt.plot(hist.history['val_loss'], 'r', label='val loss')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.legend(loc='upper right')
# plt.ylim(-0.001,2)
# acc_ax.plot(hist.history['mse'], 'b', label='train acc')
# acc_ax.plot(hist.history['val_mse'], 'g', label='val acc')
# acc_ax.set_ylabel('accuracy')
# acc_ax.legend(loc='upper left')
#
# y=model.predict_classes(x_test_data.T)
# cf = confusion_matrix(y_test_data.T, y)
# print(cf)
# y=model.predict_classes(x_train_data.T)
# cf = confusion_matrix(y_train_data.T, y)
# print(cf)
# # print(xy_data.shape)
# # reconstructed_model = load_model("modelsee.h5",compile=False)
# # # print( model.predict_classes(x_train_data_norm.T),
# # #        reconstructed_model.predict_classes(x_train_data_norm.T))
plt.show()
