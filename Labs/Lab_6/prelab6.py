import numpy as np
import matplotlib.pyplot as plt

def forward_dif(f,h,s):
    f_prime = (f(s+h)-f(s))/h
    return f_prime

def centered_dif(f,h,s):
    f_prime = (f(s+h)-f(s-h))/(2*h)
    return f_prime

#***********************************
f = lambda x: np.cos(x)
dfdx = lambda x: -np.sin(x)

h = 0.01*2.**(-np.arange(0,10))
x = np.pi/2

for_f_prime = forward_dif(f,h,x)
for_e = np.abs(dfdx(x) - for_f_prime)

cen_f_prime = centered_dif(f,h,x)
cen_e = np.abs(dfdx(x) - cen_f_prime)

plt.plot(h, np.log10(for_e))
plt.plot(h, np.log10(cen_e))
plt.legend(["forward difference", "centered difference"])
plt.title("Error from approximating f'(x) using Forward and Centered Differences \n f(x) = cos(x), x = pi/2")
plt.ylabel("log_10(e)")
plt.xlabel("h")
plt.show()

print(for_f_prime)
print(cen_f_prime)


plt.plot(-h, np.log10(for_e))
plt.plot(-h, np.log10(cen_e))
stuff = 100000000000000*(-h)**5
print(h)
print(stuff)
plt.plot(-h, stuff)
plt.show()
