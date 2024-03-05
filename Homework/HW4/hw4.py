# APPM 4600 - Homework 4
# Author: Becca Blum
# Date: 3/3/24
#*****************************************************************************
#from import 
import numpy as np
import math
from numpy.linalg import norm

def Newton(F, J, x0,tol,Nmax):

    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''
    xlist = np.zeros((Nmax+1,len(x0)))
    xlist[0] = x0

    for its in range(Nmax):
       J0 = []
       F0 = np.array([f,g(x)])

       x1 = x0 - np.linalg.solve(J,F)
       xlist[its+1]=x1

       if (norm(x1-x0) < tol*norm(x0)):
           xstar = x1
           ier =0
           return[xstar, xlist,ier, its]

       x0 = x1

    xstar = x1
    ier = 1
    return[xstar,xlist,ier,its]

# Question 1 
def q1():
    f = lambda x,y: 3*x**2 - y**2
    g = lambda x,y: 3*x*y**2 - x**3 - 1

    tol = 1e-10
    Nmax = 100
    x0 = np.array([1., 1.])
    xlist = [x0]
# a)
    for n in range(Nmax):
        x = x0[0]
        y = x0[1]
        x1 = np.array([x, y]) - np.matmul(np.array([[1./6., 1./18.],[0, 1./6.]]), np.array([f(x, y), g(x, y)]))

        xlist.append(x1) 
        iteration = n
        xstar = x1
        ier = 1
        if (norm(x1-x0) < tol*norm(x0)):
            xstar = x1
            ier = 0

        x0 = x1

    print("Number of iterations", iteration)
    print("Roots are x = " + str(xstar[0]) + ", y = " + str(xstar[1]))
    print("Error message:", ier)

# c)
    J = lambda x,y: np.array([[6*x, -2*y],[3*y**2 - 3*x**2, 6*x*y]])

  
def q2():
    F = lambda x,y: np.array([x**2 + y**2 - 4, np.e**x + y - 1])
    J = lambda x,y: np.array([[2*y, 2*y],[np.e**x, 1]])


#******************************************************************************
# Call functions 
q1()