# APPM 4600 - HW#2
# Author: Becca Blum
#***********************************
import numpy as np
import matplotlib.pyplot as plt 
import math
from fixedpt_example import fixedpt
from bisection_example import bisection

# Question 1
def Q1():
    f1 = lambda x : 2*x - 1 - np.sin(x)
    
    tol = 0.5*10**(-8)

# Question 2
def Q2():
    a = 4.82
    b = 5.2
    tol = 1*10**(-4)
    Nmax = 100
# a) 
    f2a = lambda x : (x - 5)**9
    [astar_a,ier_a] = bisection(f2a,a,b,tol,Nmax)
    print("root is: " + str(astar_a) + "\n")
    print("error message: " + str(ier_a) + "\n")
# b)
    f2b = lambda x : x**9-45*x**8+900*x**7-10500*x**6+78750*x**5-393750*x**4+1312500*x**3-2812500*x**2+3515625*x-1953125
    [astar_b,ier_b] = bisection(f2b,a,b,tol,Nmax)
    print("root is: " + str(astar_b) + "\n")
    print("error message: " + str(ier_b) + "\n")

# Question 3


# Question 4
def Q4():
# a) x_n+1 = -16 + 6*x_n + 12/x_n
    f = lambda x : -16 + 6*x + 12/x
    dfdx = lambda x : 6 - 12*(x**(-2))
    x_star = 2
    x_n = x_star
    x_new = f(x_n)
    slope = dfdx(x_n)

# b) 

# Question 5
def Q5():
    f5 = lambda x : x/4 - np.sin(2*x) - 3/4
    x0_list = [-0.9, -0.57, 1.6, 3.2, 4.5]
    tol = 0.5*10**(-10)
    Nmax = 100
    x_star_list = []
    for x0 in x0_list:
        [xstar,ier] = fixedpt(f5,x0,tol,Nmax)
        x_star_list.append(xstar)
        print(ier)
    print(x_star_list)

    x = np.linspace(-2,8,100)
    y = f5(x)
    plt.plot(x,y)
    plt.plot(x,y*0)
    plt.legend(["f(x)", "x=0, for reference"])
    plt.xlabel("x"), plt.ylabel("f(x)")
    plt.title("Plot of f(x) = x - 4sin(2x) -3")
    plt.show()

def main():
    Q2()

if __name__ == "__main__":
    main()