def eval_legendre(n, x):
    phi = [1, x]
    for i in range(1,n):
        phi.append(1/(i+1)*((2*i + 1)*x*phi[i] - i*phi[i-1]))

    return phi

def chebychev(n,x):
    T = [1, x]
    for i in range(1,n):
        T.append(2*x*T[i] - T[i-1])

    return T