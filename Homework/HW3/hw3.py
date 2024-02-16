# APPM 4600 - HW#2
# Author: Becca Blum
#***********************************
import numpy as np
import matplotlib.pyplot as plt 
import math

# Question 1


# Question 2


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
    f5 = lambda x : x - 4*np.sin(2*x) - 3


def main():
    Q4()

if __name__ == "__main__":
    main()