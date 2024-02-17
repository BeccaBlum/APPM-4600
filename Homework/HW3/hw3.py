# APPM 4600 - HW#3
# Author: Becca Blum
# Date: 2/16/2024
#***********************************
import numpy as np
import matplotlib.pyplot as plt 
from fixedpt_example import fixedpt
from bisection_example import bisection

## Question 1
def Q1():
    f1 = lambda x : 2*x - 1 - np.sin(x)
    a = 0.5
    b = 1
    tol = 10**(-8)
    Nmax = 50

    [astar,ier,count] = bisection(f1,a,b,tol,Nmax)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('number of iterations used:',count)

## Question 2
def Q2():
    a = 4.82
    b = 5.2
    tol = 10**(-4)
    Nmax = 50
# a) 
    f2a = lambda x : (x - 5)**9
    [astar_a,ier_a, count] = bisection(f2a,a,b,tol,Nmax)
    print("root is: " + str(astar_a) + "\n")
    print("error message: " + str(ier_a) + "\n")
    print(count)
# b)
    f2b = lambda x : (x**9-45*x**8+900*x**7-10500*x**6+78750*x**5
        -393750*x**4+1312500*x**3-2812500*x**2+3515625*x-1953125)
    [astar_b,ier_b, count] = bisection(f2b,a,b,tol,Nmax)
    print("root is: " + str(astar_b) + "\n")
    print("error message: " + str(ier_b) + "\n")
    print(count)

## Question 3
def Q3():
# b)
    f3 = lambda x : x**3 + x - 4
    a = 1
    b = 4
    tol = 10**(-3)
    Nmax = 50

    [astar,ier,count] = bisection(f3,a,b,tol,Nmax)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('number of iterations used:',count)

## Question 4
    # in the Overleaf document

## Question 5
def Q5():
    f5 = lambda x : 5*x/4 - np.sin(2*x) - 3/4
    x0_list = [-0.9, -0.6, 1.6, 3.2, 4.6]
    tol = 0.5*10**(-10)
    Nmax = 50
    x_star_list = []
    for x0 in x0_list:
        [xstar,ier] = fixedpt(f5,x0,tol,Nmax)
        x_star_list.append(xstar)
        print(ier)
    print(x_star_list)

    x = np.linspace(-2,8,100)
    y = x - 4*np.sin(2*x) - 3
    plt.plot(x,y)
    plt.plot(x,y*0)
    plt.plot([x_star_list[1],x_star_list[3]],[0,0],'bo')
    plt.legend(["f(x)", "x=0, for reference", "findable roots using FPI"])
    plt.xlabel("x"), plt.ylabel("f(x)")
    plt.title("Plot of f(x) = x - 4sin(2x) -3")
    plt.show()

## Main function 
def main():
    #Q1()
    Q2()
    #Q3()
    #Q5()

if __name__ == "__main__":
    main()