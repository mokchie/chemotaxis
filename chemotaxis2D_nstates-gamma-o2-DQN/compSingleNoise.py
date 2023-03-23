import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import re,os
import numpy as np
matplotlib.rcParams.update({'font.size':14,'font.family':'sans-serif'})
state_size = 2
xi = 0.04
sname = 'test-n%s-xi%s'%(state_size,xi)
cmap = cm.get_cmap('jet')
fig1, ax1 = plt.subplots(1, 1)
fig2, ax2 = plt.subplots(1, 1)
fig3, (ax31,ax32,ax33) = plt.subplots(3, 1,sharex=True)
direct = "data"
epch2=epch3=1
filename2 = sname+"-greedy-epoch-%s.data"%epch2
filename3 = sname+"-DRL-epoch-%s.data"%epch3

X2 = []
Y2 = []
Xr2 = []
Yr2 = []
Kappa2 = []
R2 = []
Time2 = []
with open(direct+'/'+filename2) as fp2:
    for line in fp2:
        t, x, y, tx, ty, nx, ny, kappa = [float(item) for item in line.strip().split()]
        #            if int(np.round(t/0.01))%20==0:
        #                ax.scatter([x,],[y,],c='r',s=5)
        Time2.append(t)
        X2.append(x)
        Y2.append(y)
        Xr2.append(x+nx/kappa)
        Yr2.append(y+ny/kappa)
        Kappa2.append(kappa)
        R2.append(y)
X3 = []
Y3 = []
Xr3 = []
Yr3 = []
Kappa3 = []
R3 = []
Time3 = []
with open(direct+'/'+filename3) as fp3:
    for line in fp3:
        t, x, y, tx, ty, nx, ny, kappa = [float(item) for item in line.strip().split()]
        #            if int(np.round(t/0.01))%20==0:
        #                ax.scatter([x,],[y,],c='r',s=5)
        Time3.append(t)
        X3.append(x)
        Y3.append(y)
        Xr3.append(x+nx/kappa)
        Yr3.append(y+ny/kappa)
        Kappa3.append(kappa)
        R3.append(y)
Time2 = np.array(Time2)
X2 = np.array(X2)
Y2 = np.array(Y2)
Xr2 = np.array(Xr2)
Yr2 = np.array(Yr2)
R2 = np.array(R2)
X3 = np.array(X3)
Y3 = np.array(Y3)
Time3 = np.array(Time3)
Xr3 = np.array(Xr3)
Yr3 = np.array(Yr3)
R3 = np.array(R3)
ax1.plot(X2, Y2, '-.', color='C0', linewidth=1)
ax2.plot(Xr2, Yr2, '-.', color='C0', linewidth=1)
ax1.scatter((X2[-1],),(Y2[-1],),s=10,c='k')

ax1.plot(X3, Y3, '-', color='C1', linewidth=1)
ax2.plot(Xr3, Yr3, '-', color='C1', linewidth=1)
ax1.scatter((X3[-1],),(Y3[-1],),s=10,c='k')

ax31.plot(Time2,Kappa2,'C0',label='greedy')
ax32.plot(Time3,Kappa3,'C1',label='DRL')
ax33.plot(Time2,R2,'C0',label='greedy')
ax33.plot(Time3,R3,'C1',label='DRL')
# ax.scatter([0,],[0,],s=10,c='r')
ax1.set_aspect('equal')
#ax1.set_xlim((-10,10))
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax2.set_aspect('equal')
#ax2.set_xlim((-35,25))
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax31.set_ylabel(r'$\kappa$')
ax32.set_ylabel(r'$\kappa$')
ax33.set_ylabel(r'$\Delta c$')
ax33.set_xlabel(r'$t$')

#ax3.set_ylim((10,30))
ax31.legend(loc='best')
ax32.legend(loc='best')
ax33.legend(loc='best')
plt.show()