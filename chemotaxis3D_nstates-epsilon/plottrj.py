import matplotlib
from mpl_toolkits import mplot3d
from matplotlib import cm
import matplotlib.pyplot as plt
import re,os
import numpy as np
import matplotlib
from swimmer import Conc_field
matplotlib.rcParams.update({'font.size':14, 'font.family':'sans-serif'})
cmap = cm.get_cmap('jet')
fig1 = plt.figure()
ax1 = plt.axes(projection='3d')
fig2,ax2 = plt.subplots(1,1)
fig3,ax3 = plt.subplots(1,1)
sname = 'sample_n4'
pattern = re.compile(sname+"-epoch-([0-9]+).data$")
reward_file = sname+"-rewards.data"
filenames = []
epochs = []
direct = "data"
for root, dirs, files in os.walk(direct):
    if root == 'data':
        for name in files:
            found = pattern.match(name)
            if found:
                epochs.append(int(found.groups()[0]))
                filenames.append(name)
                print(name)

files = sorted(zip(epochs, filenames))
conc_field = Conc_field(c0=20,k=1)
Gain  = []
Epoch = []
for epch, filename in files:
    X = []
    Y = []
    Z = []
    with open(direct+'/'+filename) as fp:
        for line in fp:
            t, rx, ry, rz, tx, ty, tz, nx, ny, nz, bx, by, bz, kappa, tau = [float(item) for item in line.strip().split()]
            #            if int(np.round(t/0.01))%20==0:
            #                ax.scatter([x,],[y,],c='r',s=5)
            X.append(rx)
            Y.append(ry)
            Z.append(rz)
    if epch % 1 == 0:
        X = np.array(X)
        Y = np.array(Y)
        Z = np.array(Z)
        x0 = 0#X[0]
        y0 = 0#Y[0]
        z0 = 0#Z[0]
        X = X-x0
        Y = Y-y0
        Z = Z-z0
        ax1.plot3D(X, Y, Z, '-', color=cmap(epch / np.max(epochs)), linewidth=1)
        ax1.scatter((X[-1],),(Y[-1],),(Z[-1],),s=10,c='k')
        Epoch.append(epch)
        Gain.append(conc_field.get_conc(X[-1],Y[-1],Z[-1])-conc_field.get_conc(X[0],Y[0],Z[0]))
ax2.plot(Epoch,Gain)
ax2.set_xlabel('epoch')
ax2.set_ylabel(r'$\Delta c$')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
Epi = []
Ret = []
Rew = []
with open(direct+'/'+reward_file) as fr:
    for line in fr:
        epi,ret,rew = [float(i) for i in line.strip().split()]
        Epi.append(epi)
        Ret.append(ret)
        Rew.append(rew)
def coarseave(lst, n):
    res = []
    r = 0
    nc = 0
    for i,v in enumerate(lst):
        r += v
        nc += 1
        if (i+1)%n== 0 or i+1==len(lst):
            res.append(r/nc)
            r = 0
            nc = 0
    return res
Epi = np.array(Epi)
Ret = np.array(Ret)
Rew = np.array(Rew)
#ax3.plot(coarseave(Epi,50),coarseave(Ret,50),'b-',label='return')
ax3.plot(coarseave(Epi,100),coarseave(Rew,100),'r-',label='accumulated reward')

#ax3.plot(Epi[9::10],Rew[9::10])
ax3.set_xlabel('epoch')
ax3.set_ylabel(r'reward')
ax3.legend(loc='best')
plt.show()