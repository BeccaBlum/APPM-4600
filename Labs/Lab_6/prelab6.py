import numpy as np
import matplotlib.pyplot as plt

def forward_dif(f,h,s):
    f_prime = (f(s+h)-f(s))/h
    return f_prime

def centered_dif(f,h,s):
    f_prime = (f(s+h)-f(s-h))/(2*h)
    return f_prime

#***********************************
f = lambda x: np.cos(x)
dfdx = lambda x: -np.sin(x)

h = 0.01*2.**(-np.arange(0,10))
x = np.pi/2

for_f_prime = forward_dif(f,h,x)

cen_f_prime = centered_dif(f,h,x)

plt.plot(h, for_f_prime)
plt.plot(h, for_f_prime)
plt.plot(h, [0]*len(h) + dfdx(x))
plt.legend(["forward difference", "centered difference", "f'(pi/2) = -sin(pi/2)"])
plt.title("Approximating f'(x) using Forward and Centered Differences \n f(x) = cos(x), x = pi/2")
plt.show()
