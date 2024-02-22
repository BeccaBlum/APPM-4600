# import libraries
import numpy as np


# define routines
def pre_bisection(f,dfdx, d2fdx2, a,b,tol,Nmax):
    '''
    Inputs:
      f,a,b       - function and endpoints of initial interval
      tol, Nmax   - bisection stops when interval length < tol
                  - or if Nmax iterations have occured
    Returns:
      astar - approximation of root
      ier   - error message
            - ier = 1 => cannot tell if there is a root in the interval
            - ier = 0 == success
            - ier = 2 => ran out of iterations
            - ier = 3 => other error ==== You can explain
    '''

    '''     first verify there is a root we can find in the interval '''
    fa = f(a); fb = f(b);
    if (fa*fb>0):
       ier = 1
       astar = a
       return [astar, ier, count]

    ''' verify end point is not a root '''
    if (fa == 0):
      astar = a
      ier =0
      return [astar, ier, count]

    if (fb ==0):
      astar = b
      ier = 0
      return [astar, ier, count]

    count = 0
    while (count < Nmax):
      c = 0.5*(a+b)
      fc = f(c)

      if abs((f(c)*d2fdx2(c))/(dfdx(c))**2) < 1:
        astar = c
        ier = "Newton"
        return [astar, ier, count]

      if (fc ==0):
        astar = c
        ier = 0
        return [astar, ier, count]

      if (fa*fc<0):
         b = c
      elif (fb*fc<0):
        a = c
        fa = fc
      else:
        astar = c
        ier = 3
        return [astar, ier, count]

      if (abs(b-a)<tol):
        astar = a
        ier =0
        return [astar, ier, count]
      
      count = count +1

    astar = a
    ier = 2
    return [astar,ier, count] 

# use routines    
f = lambda x: np.e**(x**2 + 7*x - 30) - 1
dfdx = lambda x: (2*x + 7)*np.e**(x**2 + 7*x - 30)
d2fdx2 = lambda x: (2*x + 7)*(2*x + 7)*np.e**(x**2 + 7*x - 30) + (2)*np.e**(x**2 + 7*x - 30)

a = 2
b = 4.5

Nmax = 100
tol = 1e-5

[astar,ier,count] = pre_bisection(f,dfdx, d2fdx2,a,b,tol,Nmax)
print('the approximate root is',astar)
print('the error message reads:',ier)
print('number of iterations is', count)

# Newton’s method with x0 = 4.5.

