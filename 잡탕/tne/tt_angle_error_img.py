import cv2
import numpy as np
bl=[True, False]
image = np.zeros((1000,1000,3),np.uint8)
img_list =[[0 for j in range(4)] for i in range(4)]
#image =cv2.resize(image,(0,0),None,0.5,0.5)
cnt=0
for k in bl:
	for i in bl:
		for j in bl:
			for z in bl:
				name = '0 '+str(k)+' '+str(i)+' ' +str(j)+' ' +str(z) +'.png'
				img_raw = cv2.imread(name)
				img_list[cnt//4][cnt%4]=img_raw
				cnt=cnt+1
def img_tile(img_list):
    return cv2.vconcat([cv2.hconcat(img) for img in img_list])
img_raw_1=img_tile(img_list)
cv2.imshow('img',img_raw_1)
cv2.imwrite('aaaaaaaaa.jpg',img_raw_1)
cv2.waitKey(0)

