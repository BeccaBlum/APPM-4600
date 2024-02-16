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
    
def aitken(pn):
   pn_new_list = []

   for i in range(len(pn)-2):
      pn_new = pn[i] - (pn[i+1] - pn[i])**2/(pn[i+2] - 2*pn[i+1] + pn[i])
      pn_new_list.append(pn_new)
   
   pn_accel = np.array(pn_new_list)

   return pn_accel

def order_of_convergence(pn):
   conv = []
   for i in range(len(pn)-2):
      conv.append(abs(pn[i+1] - pn[i+2])/abs(pn[i] - pn[i+2]))

   return np.array(conv)

f1 =lambda x: (10/(x+4))**(1/2)

Nmax = 100
tol = 10e-10

''' test f1 '''
x0 = 1.5
[xstar,ier, x_list] = fixedpt(f1,x0,tol,Nmax)
print('the approximate fixed point is:',xstar)
print('f1(xstar):',f1(xstar))
print('Error message reads:',ier)
print(x_list)

aitken_array = aitken(x_list)
print("\n")
print(aitken_array)

x_conv = order_of_convergence(x_list)
aitken_conv = order_of_convergence(aitken_array)

print(x_conv)
print("\n")
print(aitken_conv)