import numpy as np
from numpy import random as rand
import time
import math
from scipy import io, integrate, linalg, signal
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt

# Implementation of Broyden method. B0 can either be an approx of Jf(x0) (Bmat='fwd'),
# an approx of its inverse (Bmat='inv') or the identity (Bmat='Id')
def broyden_method_nd(f,B0,x0,tol,nmax,Bmat='Id',verb=False):

    # Initialize arrays and function value
    d = x0.shape[0];
    xn = x0; #initial guess
    rn = x0; #list of iterates
    Fn = f(xn); #function value vector
    n=0;
    nf=1;
    npn=1;

    #####################################################################
    # Create functions to apply B0 or its inverse
    if Bmat=='fwd':
        #B0 is an approximation of Jf(x0)
        # Use pivoted LU factorization to solve systems for B0. Makes lusolve O(n^2)
        lu, piv = lu_factor(B0);
        luT, pivT = lu_factor(B0.T);

        def Bapp(x): return lu_solve((lu, piv), x); #np.linalg.solve(B0,x);
        def BTapp(x): return lu_solve((luT, pivT), x) #np.linalg.solve(B0.T,x);
    elif Bmat=='inv':
        #B0 is an approximation of the inverse of Jf(x0)
        def Bapp(x): return B0 @ x;
        def BTapp(x): return B0.T @ x;
    else:
        Bmat='Id';
        #default is the identity
        def Bapp(x): return x;
        def BTapp(x): return x;
    ####################################################################
    # Define function that applies Bapp(x)+Un*Vn.T*x depending on inputs
    def Inapp(Bapp,Bmat,Un,Vn,x):
        rk=Un.shape[0];

        if Bmat=='Id':
            y=x;
        else:
            y=Bapp(x);

        if rk>0:
            y=y+Un.T@(Vn@x);

        return y;
    #####################################################################

    # Initialize low rank matrices Un and Vn
    Un = np.zeros((0,d)); Vn=Un;

    if verb:
        print("|--n--|----xn----|---|f(xn)|---|");

    while npn>tol and n<=nmax:
        if verb:
            print("|--%d--|%1.7f|%1.7f|" % (n,np.linalg.norm(xn),np.linalg.norm(Fn)));

        #Broyden step xn = xn -B_n\Fn
        dn = -Inapp(Bapp,Bmat,Un,Vn,Fn);
        # Update xn
        xn = xn + dn;
        npn=np.linalg.norm(dn);

        ###########################################################
        ###########################################################
        # Update In using only the previous I_n-1
        #(this is equivalent to the explicit update formula)
        Fn1 = f(xn);
        dFn = Fn1-Fn;
        nf+=1;
        I0rn = Inapp(Bapp,Bmat,Un,Vn,dFn); #In^{-1}*(Fn+1 - Fn)
        un = dn - I0rn;                    #un = dn - In^{-1}*dFn
        cn = dn.T @ (I0rn);                # We divide un by dn^T In^{-1}*dFn
        # The end goal is to add the rank 1 u*v' update as the next columns of
        # Vn and Un, as is done in, say, the eigendecomposition
        Vn = np.vstack((Vn,Inapp(BTapp,Bmat,Vn,Un,dn)));
        Un = np.vstack((Un,(1/cn)*un));

        n+=1;
        Fn=Fn1;
        rn = np.vstack((rn,xn));

    r=xn;

    if verb:
        if npn>tol:
            print("Broyden method failed to converge, n=%d, |F(xn)|=%1.1e\n" % (nmax,np.linalg.norm(Fn)));
        else:
            print("Broyden method converged, n=%d, |F(xn)|=%1.1e\n" % (n,np.linalg.norm(Fn)));

    return(r,rn,nf)


'''
# Plot error 
    errN = np.sum((N_xlist_a - N_xstar)**2, axis=1)
    errLN = np.sum((LN_xlist_a - LN_xstar)**2, axis=1)
    plt.plot(np.arange(N_its),np.log10(errN[0:N_its]),'b-o')
    plt.plot(np.arange(LN_its),np.log10(errLN[0:LN_its]),'r-o',label='Lazy Newton')
    plt.legend(["Newton"])
    plt.show()
    '''