import matplotlib
from mpl_toolkits import mplot3d
from matplotlib import cm
import matplotlib.pyplot as plt
import re
import numpy as np
import matplotlib
import os,sys
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)
del path
from swimmer_v2 import Conc_field
matplotlib.rcParams.update({'font.size':14, 'font.family':'sans-serif'})
cmap = cm.get_cmap('jet')
fig1,ax1 = plt.subplots(1,1)
labels = [r'DDQN $N_T=2$',r'DDQN $N_T=4$']
for ii,sname in enumerate(['sample-DDQN-n2','sample-DDQN-n4']):
    pattern = re.compile(sname+"-epoch-([0-9]+).data$")
    reward_file = sname+"-rewards.data"
    filenames = []
    epochs = []
    direct = "data"
    for root, dirs, files in os.walk(direct):
        if root == direct:
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
            Epoch.append(epch)
            Gain.append(conc_field.get_conc(X[-1],Y[-1],Z[-1])-conc_field.get_conc(X[0],Y[0],Z[0]))
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
    ax1.plot(coarseave(Epi,10),coarseave(Rew,10),'-',label=labels[ii])
#ax3.plot(Epi[9::10],Rew[9::10])
ax1.set_xlabel('epoch')
ax1.set_ylabel(r'accumulative reward')
ax1.legend(loc='best')
plt.show()
