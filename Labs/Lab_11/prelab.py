# Prelab 

def trapezoid(a, b, f, N):
    h = (b-a)/N
    
    for i in range(1,N-1):
        middle = f(a+i*h)
        
    middle_sum = sum(middle)
    F = (h/2)*(f(a) + 2*middle_sum + f(b))
    
    return F
