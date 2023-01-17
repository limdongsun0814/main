import numpy as np
import matplotlib.pyplot as plt

datasets = np.loadtxt('data_linear_test2.csv', delimiter=',',skiprows=1)
normal_y = datasets[:,-6]
normal_x = datasets[:,-5]
normal_y = np.sort(normal_y/350.0)
normal_x = np.sort(normal_x/100.0)


y = datasets[:,-6]
x = datasets[:,-5]

y = np.sort(y+15)
x = np.sort(x-40)
# fig, ax1 = plt.subplots()

# plt.plot(y,normal_y,color='red')
# plt.plot(y,normal_y,'r*')
# plt.xlabel('y relative coordinates')
# plt.ylabel('y relative coordinates Normalize')


# ax2 = ax1.twinx()
plt.plot(x,normal_x,color='green')
# plt.plot(x,normal_x,'g*')
plt.xlabel('x relative coordinates')
plt.ylabel('x relative coordinates Normalize')

plt.show()