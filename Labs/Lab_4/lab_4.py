# import libraries
import numpy as np
    
# define routines
def fixedpt(f,x0,tol,Nmax):

    ''' x0 = initial guess''' 
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''
    x_list = [x0]
    count = 0
    while (count <Nmax):
       count = count +1
       x1 = f(x0)
       x_list.append(x1)
       if (abs(x1-x0) <tol):
          xstar = x1
          ier = 0
          x_array = np.array(x_list)
          return [xstar, ier, x_array]
       x0 = x1

    xstar = x1
    ier = 1
    x_array = np.array(x_list)
    return [xstar, ier, x_array]
    

f1 =lambda x: (10/(x+4))**(1/2)

Nmax = 100
tol = 10e-10

''' test f1 '''
x0 = 1.5
[xstar,ier, x_list] = fixedpt(f1,x0,tol,Nmax)
print('the approximate fixed point is:',xstar)
print('f1(xstar):',f1(xstar))
print('Error message reads:',ier)
print(len(x_list))