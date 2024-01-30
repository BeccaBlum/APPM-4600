#import math
import numpy as np
import matplotlib.pyplot as plt

y = np.array([1,2,3])
print(type(y))

# 3.1.2

print('this is 3y',3*y)

# 3.1.3

X = np.linspace(0,2*np.pi,100)
Ya = np.sin(X)
Yb = np.cos(X)

#plt.plot(X,Ya)
#plt.plot(X,Yb)
#plt.xlabel('x')
#plt.ylabel('y')
#plt.show()

#print(len(X))

#x = np.linspace(0,2*np.pi,100)
#y = np.arange(0,101,1)

#print('the first three entries of x are', x[0:3])

w = 10**(-np.linspace(1,10,10))
print(w)

x = np.linspace(1,10,len(w))
s = 3*w

#plt.semilogy(x,w)
#plt.semilogy(x,s)
#plt.show()


