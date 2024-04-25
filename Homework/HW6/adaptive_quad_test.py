# get adaptive_quad routine and numpy from adaptive_quad.py
#from adaptive_quad import *
# get plot routines
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.special

def eval_composite_trap(M,a,b,f,t):
  x = np.linspace(a,b,M)
  h = (b-a)/(M-1)
  w = h*np.ones(M)
  w[0]=0.5*w[0]; w[M-1]=0.5*w[M-1]

  I_hat = np.sum(f(x,t)*w)
  return I_hat,x,w

def adaptive_quad(a,b,f,tol,M,method,t):
  """
  Adaptive numerical integrator for \int_a^b f(x)dx

  Input:
  a,b - interval [a,b]
  f - function to integrate
  tol - absolute accuracy goal
  M - number of quadrature nodes per bisected interval
  method - function handle for integrating on subinterval
         - eg) eval_gauss_quad, eval_composite_simpsons etc.

  Output: I - the approximate integral
          X - final adapted grid nodes
          nsplit - number of interval splits
  """
  # 1/2^50 ~ 1e-15
  maxit = 50
  left_p = np.zeros((maxit,))
  right_p = np.zeros((maxit,))
  s = np.zeros((maxit,1))
  left_p[0] = a; right_p[0] = b
  # initial approx and grid
  s[0],x,_ = method(M,a,b,f,t)
  # save grid
  X = []
  X.append(x)
  j = 1
  I = 0
  nsplit = 1
  while j < maxit:
    # get midpoint to split interval into left and right
    c = 0.5*(left_p[j-1]+right_p[j-1])
    # compute integral on left and right spilt intervals
    s1,x,_ = method(M,left_p[j-1],c,f,t); X.append(x)
    s2,x,_ = method(M,c,right_p[j-1],f,t); X.append(x)

    if np.max(np.abs(s1+s2-s[j-1])) > tol:
      left_p[j] = left_p[j-1]
      right_p[j] = 0.5*(left_p[j-1]+right_p[j-1])
      s[j] = s1
      left_p[j-1] = 0.5*(left_p[j-1]+right_p[j-1])
      s[j-1] = s2
      j = j+1
      nsplit = nsplit+1
    else:
      I = I+s1+s2
      j = j-1
      if j == 0:
        j = maxit
  return I,np.unique(X),nsplit


# specify the quadrature method
method = eval_composite_trap

# interval of integration [a,b]
a=0; bs=[10,16,24,32,40]
f = lambda x,t: x**(t-1)*np.e**(-x); I_true = 1.1455808341; labl = '$f = x^{t-1}e^{-x}$'
times = [2,4,6,8,10]
# plot curve with each t
x = np.linspace(a,bs[-1],100)
for i in range(len(times)):
  f_list = f(x,times[i])
  plt.plot(x,f_list)
plt.legend(["t=2","t=4","t=6","t=8","t=10"])
plt.xlabel("x")
plt.ylabel("y")
plt.title(labl)
plt.show()
# absolute tolerance for adaptive quad
tol = 1e-3
# machine eps in numpy
eps = np.finfo(float).eps

# number of nodes and weights, per subinterval
M = 10

# loop over quadrature orders
# compute integral with non adaptive and adaptive
# compute errors for both
for i in range(len(times)):
  b = bs[i]
  # non adaptive routine
  # Note: the _,_ are dummy vars/Python convention
  # to store uneeded returns from the routine
  print("\n\nt=",times[i])
  python_gamma = scipy.special.gamma(times[i])
  print("Python built-in gamma:", python_gamma)

  I_old,_,_ = method(M,a,b,f,times[i])
  print("\nComposite Trapezoid", I_old)

  err_trap = np.abs((I_old-python_gamma))/python_gamma
  print("Relative error:",err_trap)
  
  # adaptive routine
  I_new,X,nsplit = adaptive_quad(a,b,f,tol,M,method,times[i])
  #print(nsplit)
  print("\nAdaptive Quadrature with Composite Trapezoid:",I_new)

  err_adapt = np.abs((I_new-python_gamma))/python_gamma

  print("Relative error:",err_adapt)

  '''
  err_old = np.abs(I_old-I_true)/I_true
  err_new = np.abs(I_new-I_true)/I_true
  # clean the error for nice plots
  if err_old < eps:
    err_old = eps
  if err_new < eps:
    err_new = eps
  # save grids for M = 2
  mesh = X'''
    
'''
# plot the old and new error for each f and M
fig,ax = plt.subplots(1,2)
ax[0].semilogy(M,err_old,'ro--')
ax[0].set_ylim([1e-16,2])
ax[0].set_xlabel('$M$')
ax[0].set_title('Non-adaptive')
ax[0].set_ylabel('Relative Error')
ax[1].semilogy(M,err_new,'ro--',label=labl)
ax[1].set_ylim([1e-16,2])
ax[1].set_xlabel('$M$')
ax[1].set_title('Adaptive')
ax[1].legend()
plt.show()

# plot the adaptive mesh for M=2
fig,ax = plt.subplots(1,1)
ax.semilogy(mesh,f(mesh),'ro',label=labl)
ax.legend()
plt.show()'''