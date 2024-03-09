# APPM 4600 - Homework 4
# Author: Becca Blum
# Date: 3/3/24
#*****************************************************************************
#from import 
import numpy as np
import math
from numpy.linalg import norm
import matplotlib.pyplot as plt
from broyden import broyden_method_nd

def evalF(x0,q):
    if q == 1:
        F = np.zeros(2) 
        F[0] = 3*x0[0]**2 - x0[1]**2
        F[1] = 3*x0[0]*x0[1]**2 - x0[0]**3 - 1

    if q == 2:
        F = np.zeros(2)
        F[0] = x0[0]**2 + x0[1]**2 - 4
        F[1] = np.e**x0[0] + x0[1] - 1

    if q == 3:
        F = np.zeros(3)
        F[0] = x0[0] + np.cos(x0[0]*x0[1]*x0[2]) - 1
        F[1] = (1-x0[0])**(1/4) + x0[1] + 0.05*x0[2]**2 - 0.15*x0[2] - 1
        F[2] = -x0[0]**2 -0.1*x0[1]**2 + 0.01*x0[1] + x0[2] - 1

    return F

def evalJ(x0,q):

    if q == 1:
        J = np.array([[6*x0[0], -2*x0[1]],[3*x0[1]**2 - 3*x0[0]**2, 6*x0[0]*x0[1]]])

    if q == 2: 
        J = np.array([[2*x0[0], 2*x0[1]],[np.e**x0[0], 1]])

    if q ==3: 
        J = np.array([[1 - np.sin(x0[0]*x0[1]*x0[2])*x0[1]*x0[2], - np.sin(x0[0]*x0[1]*x0[2])*x0[0]*x0[2], - np.sin(x0[0]*x0[1]*x0[2])*x0[0]*x0[1]],
                      [-(1./4.)*(1-x0[0])**(-3./4.), 1, 0.1*x0[2] - 0.15],
                      [-2*x0[0], -0.2*x0[1] + 0.01, 1]])
    return J

def Output_message(xstar,ier,its,method):
    print("\n" + method + " Method")
    if len(xstar) == 2:
        print("Root is x = " + str(xstar[0]) + ", y = " + str(xstar[1]))
    if len(xstar) == 3:
        print("Newton, root is x = " + str(xstar[0]) + ", y = " + str(xstar[1]) + ",z = " + str(xstar[2]))         
    print("Number of iterations:", its)
    print("Error message:", ier)

def Newton(x0,tol,Nmax,q):
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    xlist = np.zeros((Nmax+1,len(x0)))
    xlist[0] = x0

    for its in range(Nmax):
        J = evalJ(x0,q)
        F = evalF(x0,q)

        x1 = x0 - np.linalg.solve(J,F)
        xlist[its+1]=x1

        if (norm(x1-x0) < tol*norm(x0)):
            xstar = x1
            ier = 0
            return[xstar, xlist,ier, its]
        x0 = x1
    xstar = x1
    ier = 1
    return[xstar,xlist,ier,its]

def LazyNewton(x0,tol,Nmax,q):
    ''' Lazy Newton = use only the inverse of the Jacobian for initial guess'''
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    xlist = np.zeros((Nmax+1,len(x0)))
    xlist[0] = x0

    J = evalJ(x0, q)
    for its in range(Nmax):

       F = evalF(x0, q)
       x1 = x0 - np.linalg.solve(J,F)
       xlist[its+1]=x1

       if (norm(x1-x0) < tol*norm(x0)):
           xstar = x1
           ier =0
           return[xstar,xlist, ier,its]
       x0 = x1
    xstar = x1
    ier = 1
    return[xstar,xlist,ier,its]

# Question 1 
def q1():
    q = 1
    tol = 1e-10
    Nmax = 100
    x0 = np.array([1., 1.])
    xlist = [x0]

# a)
    its = 0
    for n in range(Nmax):
        x1 = x0 - np.matmul(np.array([[1./6., 1./18.],[0, 1./6.]]), evalF(x0,q))

        xlist.append(x1) 
        iteration = n
        xstar = x1
        ier = 1
        if (norm(x1-x0) < tol*norm(x0)):
            xstar = x1
            ier = 0
            break
        its = its + 1
        x0 = x1

    print("\nQuestion 1 output\n")
    print("Root is x = " + str(xstar[0]) + ", y = " + str(xstar[1]))
    print("Number of iterations:", iteration)
    print("Error message:", ier)

    # Plot error 
    err = np.sum((xlist - xstar)**2, axis=1)
    plt.plot(np.arange(its),np.log10(err[0:its]),'b-o')
    plt.title('Iterations log10|r-rn|')
    plt.show()

# c)
    x0 = np.array([1., 1.])
    [N_xstar,N_xlist,ier,N_its] = Newton(x0,tol,Nmax,q)
    Output_message(xstar,ier,N_its, "Newton")

    # Plot error 
    errN = np.sum((N_xlist - N_xstar)**2, axis=1)
    plt.plot(np.arange(N_its),np.log10(errN[0:N_its]),'b-o')
    plt.title('Newton iterations log10|r-rn|')
    plt.legend(["Newton"])
    plt.show()
  
def q2():
    q = 2
    tol = 1e-14
    Nmax = 100

    def F(x):
        return np.array([x0[0]**2 + x0[1]**2 - 4, np.e**x0[0] + x0[1] - 1])

# a)
    x0 = np.array([1., 1.])
    print("\nQ2.i")
    # Newton
    [N_xstar,N_xlist_a,ier,N_its] = Newton(x0,tol,Nmax,q)
    Output_message(N_xstar,ier,N_its, "Newton")

    # LazyNewton (overflow error)
    #[LN_xstar, LN_xlist_a,ier,LN_its] =  LazyNewton(x0,tol,Nmax,q)
    #Output_message(LN_xstar,ier,LN_its, "Lazy Newton")

    # Broyden (works)

# b)
    x0 = np.array([1., -1.])
    print("\nQ2.ii")
    [N_xstar, N_xlist_b,ier,N_its] = Newton(x0,tol,Nmax,q)
    Output_message(N_xstar,ier,N_its, "Newton")

    # LazyNewton
    [LN_xstar, LN_xlist_b,ier,LN_its] =  LazyNewton(x0,tol,Nmax,q)
    Output_message(LN_xstar,ier,LN_its, "Lazy Newton")

    # Broyden (works)

# c)
    x0 = np.array([0, 0])
    #print("Q2.iii")
    # Newton (does not work: J is singular matrix)
    #[N_xstar,newton_xlist_c,ier,its] = Newton(x0,tol,Nmax,q)
    #Output_message(N_xstar,ier,its, "Newton")

    # LazyNewton (does not work: J is singular matrix)
    #[LN_xstar, LN_xlist_c,ier,its] =  LazyNewton(x0,tol,Nmax,q)
    #Output_message(LN_xstar,ier,its, "Lazy Newton")

    # Broyden (overflow error)

def q3():
    q = 3
    tol = 1e-6
    x0 = [-1, 1, 1]
    Nmax = 100
    [N_xstar,N_xlist,ier,N_its] = Newton(x0,tol,Nmax,q)
    Output_message(N_xstar,ier,N_its, "Newton")

    # Plot error 
    errN = np.sum((N_xlist - N_xstar)**2, axis=1)
    plt.plot(np.arange(N_its),np.log10(errN[0:N_its]),'b-o')
    plt.title('Newton iterations log10|r-rn|')
    plt.legend(["Newton"])
    plt.show()

#******************************************************************************
# Call functions 
q3()