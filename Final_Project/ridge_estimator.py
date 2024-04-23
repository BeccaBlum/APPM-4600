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
N = 1
deg = 5#1
x = np.linspace(-5,5,20)
f = x**2 #3*x + 2

#gamma = [0]#[.6]#np.arange(-2,3,0.1)
gamma = [.6] #np.arange(-2,3,0.1)

RSS_min = np.zeros(N)
RSS_min_index = np.zeros(N)
gammas = np.zeros(N)
RSS_list = []

for ii in range(N):
    np.random.seed(3)
    noise = np.random.normal(0,1,n)
    #print(noise)
    #y = 3*x + 2 + noise
    y = x**2 + noise

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

    #A = np.transpose([x_points**0, x_points**1])
    A = np.transpose([x_points**0, x_points**1, x_points**2, x_points**3, x_points**4, x_points**5])

    RSS = []
    for i in range(len(gamma)):
        x_star = ridge_regres(A,y_points,gamma[i], deg)
        #P = x_star[1]*x + x_star[0]
        #P_control = x_star[1]*x_control + x_star[0]
        #P_points = x_star[1]*x_points + x_star[0]

        P = x_star[5]*x**5 + x_star[4]*x**4 + x_star[3]*x**3 + x_star[2]*x**2 + x_star[1]*x + x_star[0]
        P_control = x_star[5]*x_control**5 + x_star[4]*x_control**4 + x_star[3]*x_control**3 + x_star[2]*x_control**2 + x_star[1]*x_control + x_star[0]
        P_points = x_star[5]*x_points**5 + x_star[4]*x_points**4 + x_star[3]*x_points**3 + x_star[2]*x_points**2 + x_star[1]*x_points + x_star[0]

        Err_squ = abs(P_control - y_control)**2
        Err_squ_2 = abs(P_points - y_points)**2

        RSS.append(np.sum(Err_squ)) #Residual sum of squares
        RSS2 = np.sum(Err_squ_2)
    #print(ii)
    print(RSS)
    print(RSS2)
    RSS_min_index[ii] = int(np.where(RSS == np.min(RSS))[0][0])
    RSS_min[ii] = np.min(RSS)
    #print(RSS_min[ii])
    gammas[ii] = gamma[int(RSS_min_index[ii])]
    #RSS_list = [RSS_list, np.array(RSS)]
    #plt.plot(range(len(gamma)),RSS)

'''plt.plot(x,f)
plt.plot(x,P)
plt.show()
'''
#plt.xlabel("gamma")
#plt.ylabel("Residual Sum of Squares")
#plt.show()
#print(RSS_list[0][0][0][0][0])
#print(RSS_min)
#print(gammas)

plt.scatter(gammas, RSS_min)
plt.xlabel("gamma")
plt.ylabel("Minimum Residual Sum of Squares")
#plt.title("Minimum Residual Sum of Squares with respect to values of gamma between -2 and 2 \n with 100 random error trials")
plt.show()

'''
plt.hist(gammas, bins=15)
plt.xlabel("gamma")
plt.ylabel("Number of Minimum Residual Sum of Squares")
plt.show()
'''
#for 
#plt.plot(range(len(gamma)), RSS_list)
#plt.show()



plt.plot(x,f, color='k')
plt.plot(x,P, color='y',linestyle='dashed')
plt.scatter(x_points,y_points,color='c')
plt.scatter(x_control,y_control,color='m')
plt.xlabel("x")
plt.ylabel("y")
plt.legend(["f(x)","P(x)","sample points","control points"])
plt.show()
