import cv2
import os
def createFolder(directory):
	try:
		if not os.path.exists(directory):
			os.makedirs(directory)
	except OSError:
		print('directory')

cap = cv2.VideoCapture('2.avi')
cnt=3500
dir= '0'
while cap.isOpened():
    ret, frame = cap.read()
    if cnt % 10000 ==0:
		dir = './'+str(cnt)
		createFolder(dir)
		print(dir)
    data_name='./'+str(dir)+'/'+str(cnt) + '.jpg'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imwrite(data_name, frame)
    cv2.imshow('video', frame)
    cnt=cnt+1

cap.release()
cv2.destroyAllWindows()
