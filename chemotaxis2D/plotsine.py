import pdb
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
k = 1
X = np.linspace(-2,40,1000)
def fun(x):
    return np.sin(k*x)
Y = fun(X)
plt.plot(X,Y)
lamd = 2*np.pi/k
x0 = lamd/4*3-0.1*lamd
d1 = (lamd/2*0.7)/2
d2 = lamd/2-d1
#for x0 in np.linspace(0,2*np.pi,20):
if True:

    C = deque(maxlen=4)
    X0 = X
    Y0 = np.zeros_like(X0)
    plt.plot(X0,Y0,'k--')

    plt.scatter([x0-d2,x0-2*d2,x0-3*d2],[fun(x0-d2,),fun(x0-2*d2),fun(x0-3*d2)],marker='o',color='r')
    C.append(fun(x0-3*d2))
    C.append(fun(x0-2*d2))
    C.append(fun(x0-d2))

    x = x0
    d = d2
    c = 'r'
    Dl = []
    for i in range(20):
        y = fun(x)
        C.append(y)


        if np.argmin(C)==len(C)-1:
            d = d1
            c = 'b'
        elif np.argmax(C)==len(C)-1:
            d = d2
            c = 'r'

        #pdb.set_trace()
        Dl.append(d)
        plt.scatter([x, ], [y, ], marker='o', color=c)
        x += d



plt.show()
