# APPM 4600 - HW#2
# Author: Becca Blum
#***********************************
import numpy as np
import matplotlib.pyplot as plt 
import math

## Question 4

# a)
t = np.arange(0,np.pi,np.pi/30)
Y = np.cos(t)
S = np.sum(np.dot(t,Y))

print("the sum is: " + str(S))

# b)
R = 1.2
dr = 0.1
f = 15
p = 0
theta = np.linspace(0,2*np.pi,100)

x = lambda theta,R,dr,f,p: R*(1+dr*np.sin(f*theta+p))*np.cos(theta)
y = lambda theta,R,dr,f,p: R*(1+dr*np.sin(f*theta+p))*np.sin(theta)

w = x(theta,R,dr,f,p)
v = y(theta,R,dr,f,p)

#plt.plot(theta,w)
#plt.plot(theta,v)

#plt.show()

R = 1.2
dr = 0.05
f = 2
p = 1.2

for i in range(10):
    w = x(theta,i,dr,2+i,p)
    v = y(theta,i,dr,2+i,p)
    plt.plot(theta,w)
    plt.plot(theta,v)

plt.show


