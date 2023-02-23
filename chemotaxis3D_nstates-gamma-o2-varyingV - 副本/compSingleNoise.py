import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import re,os
import numpy as np
matplotlib.rcParams.update({'font.size':14,'font.family':'sans-serif'})
state_size = 4
xi2 = 0.0
xi3 = 0.08
sname2 = 'test-n%s-xi%s'%(state_size,xi2)
sname3 = 'test-n%s-xi%s'%(state_size,xi3)
cmap = cm.get_cmap('jet')
fig2 = plt.figure()
ax2 = plt.axes(projection='3d')
fig3, (ax31,ax32,ax33) = plt.subplots(3, 1,sharex=True)
direct = "data"
for epch in range(5,8,2):
    epch2=epch3=epch
    filename2 = sname2+"-greedy-epoch-%s.data"%epch2
    filename3 = sname3+"-greedy-epoch-%s.data"%epch3

    X2 = []
    Y2 = []
    Z2 = []
    R2 = []
    Time2 = []
    Kappa2 = []
    with open(direct+'/'+filename2) as fp2:
        for line in fp2:
            t, x, y, z, tx, ty, tz, nx, ny, nz, bx, by, bz, kappa, tau = [float(item) for item in line.strip().split()]
            #            if int(np.round(t/0.01))%20==0:
            #                ax.scatter([x,],[y,],c='r',s=5)
            Time2.append(t)
            X2.append(x)
            Y2.append(y)
            Z2.append(z)
            Kappa2.append(kappa)
            R2.append(y)
    X3 = []
    Y3 = []
    Z3 = []
    Kappa3 = []
    R3 = []
    Time3 = []
    with open(direct+'/'+filename3) as fp3:
        for line in fp3:
            t, x, y, z, tx, ty, tz, nx, ny, nz, bx, by, bz, kappa, tau = [float(item) for item in line.strip().split()]
            #            if int(np.round(t/0.01))%20==0:
            #                ax.scatter([x,],[y,],c='r',s=5)
            Time3.append(t)
            X3.append(x)
            Y3.append(y)
            Z3.append(z)
            Kappa3.append(kappa)
            R3.append(y)
    Time2 = np.array(Time2)
    X2 = np.array(X2)
    Y2 = np.array(Y2)
    Z2 = np.array(Z2)
    R2 = np.array(R2)
    X3 = np.array(X3)
    Y3 = np.array(Y3)
    Z3 = np.array(Z3)
    Time3 = np.array(Time3)
    R3 = np.array(R3)
    ax2.plot3D(X2, Y2, Z2,'-.', color='C0', linewidth=1)
    ax2.scatter((X2[-1],),(Y2[-1],),(Z2[-1],),s=10,c='k')

    ax2.plot3D(X3, Y3, Z3, '-', color='C1', linewidth=1)
    ax2.scatter((X3[-1],),(Y3[-1],),(Z3[-1],),s=10,c='k')

    ax31.plot(Time2,Kappa2,'-',color='C0')
    ax32.plot(Time3,Kappa3,'-',color='C1')
    ax33.plot(Time2,R2,'C0')
    ax33.plot(Time3,R3,'C1')
# ax.scatter([0,],[0,],s=10,c='r')
#ax1.set_xlim((-10,10))
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')

ax31.set_ylabel(r'$\kappa$')
ax32.set_ylabel(r'$\kappa$')
ax33.set_ylabel(r'$c$')
ax33.set_xlabel(r'$t$')

#ax3.set_ylim((10,30))
#ax31.legend(loc='best')
#ax32.legend(loc='best')
#ax33.legend(loc='best')
plt.show()