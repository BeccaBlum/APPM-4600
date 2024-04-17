import numpy as np
import matplotlib.pyplot as plt

def ridge_regres(A,b,gamma, deg):
    s = len(A)
    leftside = np.matmul(np.transpose(A),A) + gamma*np.identity(deg+1)
    rightside = np.matmul(np.transpose(A),b)
    x_star = np.linalg.solve(leftside,rightside)

    return x_star

n = 20
k = 10
N = 300
deg = 1
x = np.linspace(0,10,20)
f = 3*x + 2

gamma = np.arange(-2,2,0.1)

RSS_min = np.zeros(N)
RSS_min_index = np.zeros(N)
gammas = np.zeros(N)
RSS_list = []

for ii in range(N):
    noise = np.random.normal(0,1,n)
    y = 3*x + 2 + .2*noise

    j = 0
    y_points = np.zeros(k); x_points = np.zeros(k)
    x_control = np.zeros(k); y_control = np.zeros(k)
    for i in range(n):
        if (i % 2) == 0:
            y_points[j] = y[i]
            x_points[j] = x[i]
        else:
            x_control[j] = x[i]
            y_control[j] = y[i]
            j = j+1

    A = np.transpose([x_points**0, x_points**1])

    RSS = []
    for i in range(len(gamma)):
        x_star = ridge_regres(A,y_points,gamma[i], deg)
        P = x_star[1]*x + x_star[0]
        P_control = x_star[1]*x_control + x_star[0]

        Err_squ = abs(P_control - y_control)**2

        RSS.append(np.sum(Err_squ)) #Residual sum of squares
    #print(ii)
    #print(RSS)
    RSS_min_index[ii] = int(np.where(RSS == np.min(RSS))[0][0])
    RSS_min[ii] = np.min(RSS)
    #print(RSS_min[ii])
    gammas[ii] = gamma[int(RSS_min_index[ii])]
    #RSS_list = [RSS_list, np.array(RSS)]
    plt.plot(range(len(gamma)),RSS)

plt.xlabel("gamma")
plt.ylabel("Residual Sum of Squares")
plt.show()
#print(RSS_list[0][0][0][0][0])
#print(RSS_min)
#print(gammas)
plt.scatter(gammas, RSS_min)
plt.xlabel("gamma")
plt.ylabel("Minimum Residual Sum of Squares")
#plt.title("Minimum Residual Sum of Squares with respect to values of gamma between -2 and 2 \n with 100 random error trials")
plt.show()

plt.hist(gammas, bins=15)
plt.xlabel("gamma")
plt.ylabel("Number of Minimum Residual Sum of Squares")
plt.show()

#for 
#plt.plot(range(len(gamma)), RSS_list)
#plt.show()

'''
plt.plot(x,P)
plt.scatter(x_points,y_points)
plt.scatter(x_control,y_control)
plt.plot(x,f)
plt.show()
'''