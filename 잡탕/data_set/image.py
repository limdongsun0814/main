import cv2
import os
import numpy as np
import matplotlib.pylab as plt
def createFolder(directory):
	try:
		if not os.path.exists(directory):
			os.makedirs(directory)
	except OSError:
		print('directory')
my_robot_img = []
video_name=[]# 50,0.avi
old_data = 0
x_data=[]
my_robot_hist_data=[]
for i in range(50,241,10):
    name= 'data/'+str(i)+','+'0.avi'
    print(name)
    video_name.append(name)
for i in video_name:
    cap = cv2.VideoCapture(i)
    cnt=3500
    dir= '0'
    x=[]
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


        img_resize = cv2.resize(frame, (600, 400), interpolation=cv2.INTER_LINEAR)
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


        my_robot_img.append(frame)
        cv2.imshow('video', frame)
    for i in range(len(x)):
        for j in range(len(x[i])):
            x_sum=x[i][j]+x_sum
        x_avg.append( x_sum / len(x[i]))
        # print(x_sum / len(x[i]))
    # print(len(x_avg))
    # x_data.append(x_avg)
    x_data.append(np.average(x,axis=0).tolist())
    # print(len(x_data[0]))
    cap.release()
    cv2.destroyAllWindows()

x=[]


for i in range(len(my_robot_img)):
    print(i)

    img_resize = cv2.resize(my_robot_img[i], (600, 400), interpolation=cv2.INTER_LINEAR)
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
    # print(data)
    x.append(data)





# y_arr=np.array(y)
# print(x)
x_arr= np.array(x)
# x_arr= np.array(x_data)
# print(y_arr.shape)
print(x_arr.shape)
# print(x_arr.shape)
# print(np.max(x_arr[:, 0]))
# bin=np.arange(0,np.max(x_arr[:, 0])+10,(np.max(x_arr[:, 0])+10)/20)
# hist,bins=np.histogram(x_arr[:, 0],bins=bin)
# print(len(hist))
# print(bin)
plt.figure(1)
plt.pcolor(x_arr[:,:600])#,bins=bin)
plt.colorbar()
# plt.yticks(np.arange(0, 13), np.arange(50,221,10))
plt.xlabel('x-axis count data')
plt.ylabel('distance')

plt.figure(2)
plt.pcolor(x_arr[:,600:1000])#,bins=bin)
plt.colorbar()
# plt.yticks(np.arange(0, 13), np.arange(50,221,10))
plt.xlabel('y-axis count data')
plt.ylabel('distance')

plt.show()