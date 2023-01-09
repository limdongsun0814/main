import numpy as np
import math
import csv
import os
import copy
import matplotlib.pyplot as plt
import tensorflow as tf

tf.disable_v2_behavior()
#print(tf.test.is_built_with_cuda())
#print(tf.test.is_gpu_available())
f = open('focus 330nm.csv', 'r')
print(1/2*3)
rdr = csv.reader(f)
y_train = []
x_train_list = []
x_train_list_all=[]
y_train_all=[]
z_train_list_all,z_train_list=[],[]
plx = 41.41 / 480
for line in rdr:
    #if float(line[1]) < 200:
    y_train.append(float(line[1]))
    x_train_list.append(float(line[0]))
    x_train_list_all.append(float(line[0]))
    y_train_all.append(float(line[1]))
x_train_all = np.array(x_train_list_all)
x_train = np.array(x_train_list)
y_train_array=np.array(y_train)
x=[1.0,1.0,1.0,1.0]#constant(tf.random.normal([1], 0, 1, tf.float32)
#a:  [-0.0241803] b:  [0.07064048] c:  [0.23725522] d:  [-0.6547566] e:  [-0.4973882] f:  [0.97130126] g:  [0.9754795]
sw = 1
w_old=0.0
learning_rate = 0.00001
if sw==1:
    w = tf.Variable(tf.random.normal([1]), name='weight')#
    a = tf.Variable(tf.random.normal([1]),name='a')
    b = tf.Variable(tf.random.normal([1]),name='b')#
    c = tf.Variable(tf.random.normal([1]),name='c')
    d = tf.Variable(tf.random.normal([1]),name='d')#
    e = tf.Variable(tf.random.normal([1]),name='e')#
    f = tf.Variable(tf.random.normal([1]),name='f')#
    g = tf.Variable(tf.random.normal([1]),name='g')
else:
    csv_r = open('save.csv', 'r')
    rdr = csv.reader(csv_r)
    for line in rdr:
        print(line[0])
        w = tf.Variable(float(line[0]), name='weight')  #
        a = tf.Variable(float(line[1]), name='a')
        b = tf.Variable(float(line[2]), name='b')  #
        c = tf.Variable(float(line[3]), name='c')
        d = tf.Variable(float(line[4]), name='d')  #
        e = tf.Variable(float(line[5]), name='e')  #
        f = tf.Variable(float(line[6]), name='f')  #
        g = tf.Variable(float(line[7]), name='g')
        csv_r.close()
        break


#hypothesis = w/(tf.tan(tf.atan(tf.tan(b*math.pi/180)/480)-tf.atan((a-x_train)*tf.tan(b*math.pi/180)/480))+tf.tan(tf.atan((a-x_train)*tf.tan(b*math.pi/180)/480)))
#hypothesis = a*(x_train+w)**6+b*(x_train+w)**5+c*(x_train+w)**4+d*(x_train+w)**3+e*(x_train+w)**2+f*(x_train+w)+g
hypothesis = tf.math.sqrt(a*(x_train)+b)**c+d#+tf.numpy.heaviside(x_train-f,1)*(w*x_train**2)#+f*x_train**2

cost = tf.reduce_mean(tf.square((hypothesis - y_train)**2))

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
step = 0
old_cost = 0
aa,bb,cc,dd,ww=0,0,0,0,0
min_cost = 100
while sess.run(cost) > 0.00000000001 or np.isnan(sess.run(cost)):
    sess.run(train)
    if step % 100 == 0:

        print(step, "cost: ", sess.run(cost),"a: ", sess.run(a),"b: ",  sess.run(b),"c: ", sess.run(c),"d: ", sess.run(d),"e: ", sess.run(e),"f: ", sess.run(f),"g: ", sess.run(g))#,"d: ", sess.run(d))# sess.run(cost), sess.run(w), sess.run(a), sess.run(b), sess.run(c))
        #print(sess)
        #hypothesis_np=sess.numpy()
        if step % 10000 == 0:
            aa=sess.run(a)
            bb=sess.run(b)
            cc=sess.run(c)
            dd=sess.run(d)
            ee=sess.run(e)
            ff=sess.run(f)
            gg=sess.run(g)

            # val = bb + abs(ww) / (np.tan(((dd - x_train_all) * -48.8 / 480) * math.pi / 180 + (
            #        z_train_np_all * (48.8 / 480)) * math.pi / 180) + np.tan(
            #    ((dd - x_train_all) * 48.8 / 480) * math.pi / 180))

            val =np.sqrt(aa*(x_train_all)+bb)**cc+dd#+np.heaviside(x_train_all-ff,1)*(ww*x_train**2)#+ff*x_train_all**2

            # val = bb+abs(ww)/(np.tan((dd-(x_train_all+z_train_np_all)*aa)*(48.8/480)*math.pi/180+(z_train_np_all*(48.8/480))*math.pi/180)+np.tan((dd-(x_train_all+z_train_np_all)*aa)*(48.8/480)*math.pi/180))
            plt.clf()
            plt.subplot(2, 1, 1)
            plt.plot(val - y_train_all, '*')

            plt.subplot(2, 1, 2)
            plt.plot(val, '*')

            plt.subplot(2, 1, 2)
            plt.plot(y_train_all, '*')
            # plt.ylim(-20,20)
            plt.show(block=False)
            plt.pause(0.0000000000000000000000000001)
            print("=================================================")


        #if old_cost == sess.run(cost):
            #sess.run(tf.global_variables_initializer())
            #step = 0
            #old_cost=0
        #else:
            #old_cost=sess.run(cost)
    step += 1
    if np.isnan(sess.run(cost)) and np.isnan(sess.run(w)) and step > 100:  # and np.isnan(sess.run(b)):
        print(sess.run(cost),sess.run(w))
        sess.run(tf.global_variables_initializer())
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++")
        step = 0
    if np.isnan(sess.run(cost)):  # and np.isnan(sess.run(b)):
        print(sess.run(cost),sess.run(w))
        sess.run(tf.global_variables_initializer())
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++")
        step = 0

print("cost: ", sess.run(cost),"w: ", sess.run(w),"a: ",  sess.run(a),"b: ", sess.run(b),"c: ", sess.run(c),"d: ", sess.run(d))#,"d: ", sess.run(d), )  # "b: ",sess.run(b),
print("=================================================")
print(x)
