import numpy as np
#from interp import *
import matplotlib.pyplot as plt

def monomial_exp(f, degree, Neval):
    xint = np.zeros(degree)
    yint = np.zeros(degree)
    Pz = np.zeros(Neval)
    h = 2/(degree-1)
    for j in range(degree):
        xint[j] = -1 + (j - 1)*h
        yint[j] = f(xint[j])

    print(len(xint))
    V = np.vander(xint)
    a_coeff = np.linalg.solve(V,yint)
    #print(a_coeff) 

    return a_coeff

#**************************************************
f = lambda x: 1/(1 + (10*x)**2)
interval = [-1, 1]
N = 1000
degree = 3
a_coeff = monomial_exp(f, degree, N)

plt.plot(z,P)
plt.show()