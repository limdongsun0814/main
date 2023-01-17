import tensorflow as tf
import numpy as np
import math
import csv
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import butter, lfilter, freqz

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


f = open('distance.csv','r')
w = open('distance_filter.csv','w')
wt=csv.writer(w)
rdr = csv.reader(f)
y=np.array([])

for i in rdr:
    for j in i:
        #for z in range(10):
        y=np.append(y,float(j))
        #print(type(j))

print(y[0])
#y_filter=[]
#for i in y:
#    y_filter.append(filter.filter(float(i)))
#y_filter=filter.filter(y)
#y_filter_fft=abs(np.fft.fft(y_filter)/len(y_filter))
#y_fft=abs(np.fft.fft(y)/len(y))

y_filter=butter_lowpass_filter(y,cutoff=0.1,fs=5,order=1)

for i in y_filter:
    wt.writerow([i])
#plt.scatter(range(len(y)),y)
plt.subplot(4, 1, 1)
#plt.xlim(0,500)
#plt.ylim(100,150)
#plt.plot(y_fft)
plt.plot(y)

plt.subplot(4, 1, 2)
#plt.xlim(0,500)
#plt.ylim(100,150)
plt.plot(y_filter)
plt.plot(y)

y_fft=abs(np.fft.fft(y)/len(y))
plt.subplot(4,1,3)
#plt.xlim(0,500)
#plt.ylim(100,150)
plt.plot(y_fft)

y_filter_fft=abs(np.fft.fft(y_filter/len(y_filter)))
plt.subplot(4,1,4)
#plt.xlim(0,500)
#plt.ylim(100,150)
plt.plot(y_filter_fft)
plt.show()

