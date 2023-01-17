import matplotlib.pyplot as plt
import numpy as np
import time
a=np.array([0])
while True:
    plt.plot(a,'*r')
    plt.show(block=False)
    plt.pause(0.01)
    #time.sleep(10)
    #plt.close()
    a[0]=a[0]+1
