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
video_name=['200,0.avi','210,0.avi','220,0.avi']
my_robot_img = []
for i in video_name:
    cap = cv2.VideoCapture(i)
    cnt=3500
    dir= '0'
    while cap.isOpened():
        ret, frame = cap.read()
        print(None==frame)
        print('type',frame is None)#,type(frame),len(frame))
        # if   all(frame == None)==True:
        #     break
        if (cv2.waitKey(1) & 0xFF == ord('q')) or (frame is None):
            print('stop')
            break
        my_robot_img.append(frame)
        cv2.imshow('video', frame)
        cnt=cnt+1

    cap.release()
    cv2.destroyAllWindows()
my_robot_hist_data=[]
x=[]



for i in range(len(my_robot_img)):
    # print(i)

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
    print(data)
    x.append(data)





# y_arr=np.array(y)
print(x)
x_arr= np.array(x)
# print(y_arr.shape)
print(x_arr.shape)
# print(x_arr.shape)
# print(np.max(x_arr[:, 0]))
# bin=np.arange(0,np.max(x_arr[:, 0])+10,(np.max(x_arr[:, 0])+10)/20)
# hist,bins=np.histogram(x_arr[:, 0],bins=bin)
# print(len(hist))
# print(bin)
plt.pcolor(x_arr)#,bins=bin)
plt.colorbar()
plt.show()