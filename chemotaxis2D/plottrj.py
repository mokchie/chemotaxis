import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import re,os
import numpy as np
from swimmer import Conc_field

matplotlib.rcParams.update({'font.size':14,'font.family':'sans-serif'})
conc_field = Conc_field(c0=20,k=1)
cmap = cm.get_cmap('jet')
state_size = 8
sname = 'test-%s'%state_size
fig0, ax0 = plt.subplots(1, )
fig1,(ax1,ax2,ax3) = plt.subplots(3, 1, sharex=True)
fig2,ax4 = plt.subplots(1, 1, sharex=True)
axs = (ax1,ax2,ax3)
filenames = [sname+'-1-epoch-10.data',sname+'-2-epoch-10.data',sname+'-3-epoch-10.data']
labels = ['swinging','greedy','DRL']
colors = ['C0','C1','C2']
direct = "data"
k0 = 4.0
Taction = 1/state_size
v0 = 1
dt = 0.01
ntimes = int(Taction * 2 * np.pi / k0 / v0 / dt)
for j,filename in enumerate(filenames):
    X = []
    Y = []
    Xr = []
    Yr = []
    kappa0 = -1
    Kappa = []
    Time = []
    Ta = []
    Ca = []
    C = []
    jj = 0
    with open(direct+'/'+filename) as fp:
        for line in fp:
            t, x, y, nx, ny, kappa = [float(item) for item in line.strip().split()]
            #            if int(np.round(t/0.01))%20==0:
            #                ax.scatter([x,],[y,],c='r',s=5)
            X.append(x)
            Y.append(y)
            C.append(conc_field.get_conc(x,y))
            xr = x+nx/kappa
            Xr.append(xr)
            yr = y+ny/kappa
            Yr.append(yr)
            Time.append(t)
            Kappa.append(kappa)
            #if np.abs(kappa0-kappa)>1e-4 and jj>0:
            #    pass
            #    ax0.scatter((x0,),(y0,),c='b')
            if jj%ntimes==0:
                Ta.append(t)
                Ca.append(conc_field.get_conc(x,y))
                #ax.plot((x,xr),(y,yr),color='b')
            kappa0 = kappa
            x0 = x
            y0 = y
            jj += 1
    ax0.plot(X, Y, '-', color=colors[j], label=labels[j], linewidth=1)
    #ax0.plot(Xr, Yr, '--', color=colors[j], linewidth=1)
    #ax.scatter((X[-1],),(Y[-1],),s=10,c='k')
    axs[j].plot(Time,Kappa,color=colors[j])
    ax4.plot(Time,C,color=colors[j],label=labels[j])
    #axs[j+1].scatter(Ta,Ca,s=4,c='r')

# ax.scatter([0,],[0,],s=10,c='r')
ax0.set_aspect('equal')
ax0.set_xlabel('$x$')
ax0.set_ylabel('$y$')
ax0.legend(loc = 'upper right')
#ax0.set_xlim((-1,22))
ax4.legend(loc = 'upper left')
#ax0.set_xlim((-10,-4))
#ax1.set_xlabel('$t$')
ax1.set_ylabel(r'$\kappa$')
ax4.set_xlabel('$t$')
ax4.set_ylabel(r'$c$')
#ax3.set_xlabel('$t$')
ax2.set_ylabel(r'$\kappa$')
#ax4.set_xlabel('$t$')
ax3.set_xlabel('$t$')
ax3.set_ylabel(r'$\kappa$')
#ax246.set_ylim((20,50))
plt.show()
