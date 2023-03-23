import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import re,os
import numpy as np
matplotlib.rcParams.update({'font.size':14,'font.family':'sans-serif'})
state_size = 2
xi = 0.06
sname = 'test-n%s-xi%s'%(state_size,xi)
cmap = cm.get_cmap('jet')
fig1, ax1 = plt.subplots(1, 1)
fig2, ax2 = plt.subplots(1, 1)
fig3, ax3 = plt.subplots(1, 1)
pattern2 = re.compile(sname+"-greedy-epoch-([0-9]+).data$")
pattern3 = re.compile(sname+"-DRL-epoch-([0-9]+).data$")
filenames2 = []
filenames3 = []
epochs2 = []
epochs3 = []
direct = "data"
for root, dirs, files in os.walk(direct):
    if root == direct:
        for name in files:
            found2 = pattern2.match(name)
            if found2:
                epochs2.append(int(found2.groups()[0]))
                filenames2.append(name)
                print(name)
            found3 = pattern3.match(name)
            if found3:
                epochs3.append(int(found3.groups()[0]))
                filenames3.append(name)
                print(name)

R2 = []
R3 = []
files = sorted(zip(epochs2, filenames2, epochs3, filenames3))
for epch2, filename2, epch3, filename3 in files:
    X2 = []
    Y2 = []
    Xr2 = []
    Yr2 = []
    with open(direct+'/'+filename2) as fp2:
        for line in fp2:
            t, x, y, tx, ty, nx, ny, kappa = [float(item) for item in line.strip().split()]
            #            if int(np.round(t/0.01))%20==0:
            #                ax.scatter([x,],[y,],c='r',s=5)
            X2.append(x)
            Y2.append(y)
            Xr2.append(x+nx/kappa)
            Yr2.append(y+ny/kappa)
    X3 = []
    Y3 = []
    Xr3 = []
    Yr3 = []
    with open(direct+'/'+filename3) as fp3:
        for line in fp3:
            t, x, y, tx, ty, nx, ny, kappa = [float(item) for item in line.strip().split()]
            #            if int(np.round(t/0.01))%20==0:
            #                ax.scatter([x,],[y,],c='r',s=5)
            X3.append(x)
            Y3.append(y)
            Xr3.append(x+nx/kappa)
            Yr3.append(y+ny/kappa)
    if epch2 % 1 == 0:

        X2 = np.array(X2)
        Y2 = np.array(Y2)

        Xr2 = np.array(Xr2)
        Yr2 = np.array(Yr2)

        X3 = np.array(X3)
        Y3 = np.array(Y3)

        Xr3 = np.array(Xr3)
        Yr3 = np.array(Yr3)
        if epch2%4==0:
            ax1.plot(X2, Y2, '-.', color=cmap(epch2 / np.max(epochs2)), linewidth=1)
            ax2.plot(Xr2, Yr2, '-.', color=cmap(epch2 / np.max(epochs2)), linewidth=1)
            ax1.scatter((X2[-1],),(Y2[-1],),s=10,c='k')

            ax1.plot(X3, Y3, '-', color=cmap(epch3 / np.max(epochs3)), linewidth=1)
            ax2.plot(Xr3, Yr3, '-', color=cmap(epch3 / np.max(epochs3)), linewidth=1)
            ax1.scatter((X3[-1],),(Y3[-1],),s=10,c='k')

        R2.append(Y2[-1]-Y2[0])
        R3.append(Y3[-1] - Y3[0])
ax3.plot(np.array(range(len(R2)))+1,R2,'v--',color='C2',label='greedy')
ax3.plot(np.array(range(len(R2)))+1,np.average(R2)+np.zeros_like(R2),'-',color='C2')
ax3.plot(np.array(range(len(R3)))+1,R3,'v--',color='C3',label='DRL')
ax3.plot(np.array(range(len(R3)))+1,np.average(R3)+np.zeros_like(R3),'-',color='C3')
# ax.scatter([0,],[0,],s=10,c='r')
ax1.set_aspect('equal')
#ax1.set_xlim((-10,10))
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax2.set_aspect('equal')
#ax2.set_xlim((-35,25))
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax3.set_xlabel(r'$i$')
ax3.set_ylabel(r'$\Delta c$')
#ax3.set_ylim((10,30))
ax3.legend(loc='best')
plt.show()