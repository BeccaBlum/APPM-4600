import numpy as np
import numpy.linalg as la 
import math
from datetime import datetime

def driver():
    n = 3 #100
    #x = np.linspace(0,np.pi,n)

    #f = lambda x: x**2 + 4*x +2*np.exp(x)
    #g = lambda x: 6*x**3 + 2*np.sin(x)

    #y = f(x)
    #w = g(x)
    y = np.array([1, 0, 0])
    w = np.array([1, 1, 0])
    dp = dotProduct(y,w,n)

    print('the dot product is: ', dp)

    A = np.array([[1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8]])
    v = np.array([1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8])
 
    v = np.linspace(0,100,1000)

    dt = datetime.now()
    t0 = dt.microsecond
    #C = matrixVecMul(A,v)
    C = dotProduct(v,v,len(v))
    dt = datetime.now()
    t1 = dt.microsecond
    total = t1-t0
    print(total)

    dt = datetime.now()
    t0 = dt.microsecond
    #C2 = np.matmul(A,v)
    C2 = np.dot(v,v)
    dt = datetime.now()
    t1 = dt.microsecond
    total = t1-t0
    print(total)

    print(C)
    print(C2)

    return

def dotProduct(x, y, n):
    dp = 0.
    for j in range(n):
        dp = dp +x[j]*y[j]
        
    return dp 

def matrixVecMul(A,v):  
    C = np.zeros(len(A))

    for i in range(len(A)):        
        for j in range(len(A[0])): 
            C[i] = C[i] + A[i,j] * v[j]

    return C


driver()