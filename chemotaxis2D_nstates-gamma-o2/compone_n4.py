import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import re,os
import numpy as np
matplotlib.rcParams.update({'font.size':14,'font.family':'sans-serif'})
state_size = 4
sname = 'test-DDQN-n%s'%(state_size,)
cmap = cm.get_cmap('jet')
fig1, ax1 = plt.subplots(1, 1)
axin = ax1.inset_axes((0.02,0.02,0.43,0.43))
fig2, ax2 = plt.subplots(1, 1)
fig3, (ax31,ax32,ax33) = plt.subplots(3, 1,sharex=True)
N = 1
filenames1 = [sname+"-swinging-epoch-%s.data"%N,]
filenames2 = [sname+"-greedy-epoch-%s.data"%N,]
filenames3 = [sname+"-DRL-epoch-%s.data"%N,]
epochs1 = [N,]
epochs2 = [N,]
epochs3 = [N,]
direct = "data"
R1 = []
R2 = []
R3 = []
files = sorted(zip(epochs1, filenames1, epochs2, filenames2, epochs3, filenames3))
for epch1, filename1, epch2, filename2, epch3, filename3 in files:
    X1 = []
    Y1 = []
    Xr1 = []
    Yr1 = []
    Time1 = []
    Kappa1 = []
    with open(direct+'/'+filename1) as fp1:
        for line in fp1:
            t, x, y, tx, ty, nx, ny, kappa = [float(item) for item in line.strip().split()]
            #            if int(np.round(t/0.01))%20==0:
            #                ax.scatter([x,],[y,],c='r',s=5)
            Time1.append(t)
            Kappa1.append(kappa)
            X1.append(x)
            Y1.append(y)
            Xr1.append(x+nx/kappa)
            Yr1.append(y+ny/kappa)
    X2 = []
    Y2 = []
    Xr2 = []
    Yr2 = []
    Time2 = []
    Kappa2 = []
    with open(direct+'/'+filename2) as fp2:
        for line in fp2:
            t, x, y, tx, ty, nx, ny, kappa = [float(item) for item in line.strip().split()]
            #            if int(np.round(t/0.01))%20==0:
            #                ax.scatter([x,],[y,],c='r',s=5)
            Time2.append(t)
            X2.append(x)
            Y2.append(y)
            Kappa2.append(kappa)
            Xr2.append(x+nx/kappa)
            Yr2.append(y+ny/kappa)
    X3 = []
    Y3 = []
    Xr3 = []
    Yr3 = []
    Time3 = []
    Kappa3 = []
    with open(direct+'/'+filename3) as fp3:
        for line in fp3:
            t, x, y, tx, ty, nx, ny, kappa = [float(item) for item in line.strip().split()]
            #            if int(np.round(t/0.01))%20==0:
            #                ax.scatter([x,],[y,],c='r',s=5)
            X3.append(x)
            Y3.append(y)
            Time3.append(t)
            Kappa3.append(kappa)
            Xr3.append(x+nx/kappa)
            Yr3.append(y+ny/kappa)
    if epch1 % 1 == 0:
        Time1 = np.array(Time1)
        X1 = np.array(X1)
        Y1 = np.array(Y1)

        Xr1 = np.array(Xr1)
        Yr1 = np.array(Yr1)

        Kappa1 = np.array(Kappa1)

        Time2 = np.array(Time2)
        X2 = np.array(X2)
        Y2 = np.array(Y2)

        Xr2 = np.array(Xr2)
        Yr2 = np.array(Yr2)

        Kappa2 = np.array(Kappa2)
        Time3 = np.array(Time3)
        X3 = np.array(X3)
        Y3 = np.array(Y3)

        Xr3 = np.array(Xr3)
        Yr3 = np.array(Yr3)

        Kappa3 = np.array(Kappa3)

        if epch1%1==0:
            ax1.plot(X1, Y1, '-', color='C0', linewidth=1,label='alternating')
            axin.plot(X1, Y1, '-', color='C0', linewidth=1)
            #ax2.plot(Xr1, Yr1, '-', color='C0', linewidth=1)
            ax2.plot(Time1,Y1-Y1[0], '-', color='C0', linewidth=1,label='alternating')            
            ax1.scatter((X1[0],), (Y1[0],), s=10, c='r', zorder=10)
            axin.scatter((X1[0],), (Y1[0],), s=10, c='r', zorder=10)
            ax1.scatter((X1[-1],), (Y1[-1],), s=10, c='k',zorder=10)
            ax31.plot(Time1,Kappa1,'-',color='C0',linewidth=1)

            ax1.plot(X2, Y2, '-', color='C1', linewidth=1,label='short-sighted')
            axin.plot(X2, Y2, '-', color='C1', linewidth=1)
            #ax2.plot(Xr2, Yr2, '-', color='C1', linewidth=1)
            ax2.plot(Time2,Y2-Y2[0], '-', color='C1', linewidth=1,label='short-sighted')                        
            ax1.scatter((X2[-1],),(Y2[-1],),s=10,c='k',zorder=10)
            ax32.plot(Time2, Kappa2, '-', color='C1', linewidth=1)

            ax1.plot(X3, Y3, '-', color='C2', linewidth=1,label='DRL')
            axin.plot(X3, Y3, '-', color='C2', linewidth=1)
            #ax2.plot(Xr3, Yr3, '-', color='C2',linewidth=1)
            ax2.plot(Time3,Y3-Y3[0], '-', color='C2', linewidth=1,label='DRL')
            ax1.scatter((X3[-1],),(Y3[-1],),s=10,c='k',zorder=10)
            ax33.plot(Time3, Kappa3, '-', color='C2', linewidth=1)

        R1.append(Y1[-1]-Y1[0])
        R2.append(Y2[-1]-Y2[0])
        R3.append(Y3[-1]-Y3[0])
# ax.scatter([0,],[0,],s=10,c='r')
ax1.set_aspect('equal')
#ax1.set_xlim((-10,10))
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax2.set_xlabel(r'$t$')
ax2.set_ylabel(r'$\Delta c/k_c$')
ax2.legend(loc='best')
ax33.set_xlabel(r'$t$')
ax31.set_ylabel(r'$\kappa$')
ax32.set_ylabel(r'$\kappa$')
ax33.set_ylabel(r'$\kappa$')
ax1.legend(loc='best')
ax1.set_xlim((25,47))
axin.set_xlim((41,44))
axin.set_ylim((44,47))
ax31.set_xlim((0,80))
ax1.indicate_inset_zoom(axin,edgecolor='black')
axin.set_xticklabels([])
axin.set_yticklabels([])
fig3.set_size_inches(10, 2)
#ax3.set_ylim((10,30))
plt.show()
