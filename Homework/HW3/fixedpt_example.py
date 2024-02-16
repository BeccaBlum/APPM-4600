# import libraries
import numpy as np
    
# define routines
def fixedpt(f,x0,tol,Nmax):

    ''' x0 = initial guess''' 
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''

    count = 0
    while (count <Nmax):
       count = count +1
       x1 = f(x0)
       if (abs(x1-x0) <tol):
          xstar = x1
          ier = 0
          return [xstar,ier]
       x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier]

# use routines 
#f1 = lambda x: 1+0.5*np.sin(x)

#Nmax = 100
#tol = 1e-10

#''' test f1 '''
#x0 = 1#7**(1/5)
#[xstar,ier] = fixedpt(f1,x0,tol,Nmax)
#print('the approximate fixed point is:',xstar)
#print('f1(xstar):',f1(xstar))
#print('Error message reads:',ier)