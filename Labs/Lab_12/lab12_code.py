import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy.linalg as scila
import time


def driver():

     ''' create  matrix for testing different ways of solving a square 
     linear system'''

     '''' N = size of system'''
     N = 100
 
     ''' Right hand side'''
     b = np.random.rand(N,1)
     A = np.random.rand(N,N)
  
     x = scila.solve(A,b)
     
     test = np.matmul(A,x)
     r = la.norm(test-b)
     
     print(r)

     N = [100, 500, 1000, 2000, 4000, 5000]
     for n in N:
          print("\nN =", n)
          b = np.random.rand(n,1)
          A = np.random.rand(n,n)
          x_LU = solve_LU(A,b)

          t1 = time.perf_counter()
          x = scila.solve(A,b)
          t2 = time.perf_counter()
          built_in_solve_time = t2 - t1
          print("Built in solve time is",built_in_solve_time)


     ''' Create an ill-conditioned rectangular matrix '''
     N = 10
     M = 5
     A = create_rect(N,M)     
     b = np.random.rand(N,1)


     
def create_rect(N,M):
     ''' this subroutine creates an ill-conditioned rectangular matrix'''
     a = np.linspace(1,10,M)
     d = 10**(-a)
     
     D2 = np.zeros((N,M))
     for j in range(0,M):
        D2[j,j] = d[j]
     
     '''' create matrices needed to manufacture the low rank matrix'''
     A = np.random.rand(N,N)
     Q1, R = la.qr(A)
     test = np.matmul(Q1,R)
     A =    np.random.rand(M,M)
     Q2,R = la.qr(A)
     test = np.matmul(Q2,R)
     
     B = np.matmul(Q1,D2)
     B = np.matmul(B,Q2)
     return B     

def solve_LU(M, b):
     t1 = time.perf_counter()
     lu, piv = scila.lu_factor(M)
     t2 = time.perf_counter()
     LU_factorization_time = t2 - t1
     print("LU factorization time is", LU_factorization_time)
     
     t3 = time.perf_counter()
     x = scila.lu_solve((lu, piv), b, trans = 0)
     t4 = time.perf_counter()
     LU_solve_time = t4 - t3
     print("LU solve time is", LU_solve_time)

     total_LU_solve_time = LU_solve_time + LU_factorization_time
     print("Total LU solve time is", total_LU_solve_time)
     
     return x 
          
  
if __name__ == '__main__':
      # run the drivers only if this is called from the command line
      driver()       
