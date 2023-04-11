import os,sys
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)
del path
import matplotlib
from mpl_toolkits import mplot3d
from matplotlib import cm
import matplotlib.pyplot as plt
import re
import numpy as np
import matplotlib
from swimmer_v2 import Conc_field
matplotlib.rcParams.update({'font.size':14, 'font.family':'sans-serif'})
cmap = cm.get_cmap('jet')
fig1,ax1 = plt.subplots(1,1)
fig2,ax2 = plt.subplots(1,1)
fig3,ax3 = plt.subplots(1,1)
state_size = 4
sname = 'sample-DDQN-n%s'%state_size
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
    with open(direct+'/'+filename) as fp:
        for line in fp:
            t, rx, ry, tx, ty, nx, ny, kappa = [float(item) for item in line.strip().split()]
            #            if int(np.round(t/0.01))%20==0:
            #                ax.scatter([x,],[y,],c='r',s=5)
            X.append(rx)
            Y.append(ry)
    if epch % 1 == 0:
        X = np.array(X)
        Y = np.array(Y)
        x0 = 0#X[0]
        y0 = 0#Y[0]
        X = X-x0
        Y = Y-y0
        ax1.plot(X, Y, '-', color=cmap(epch / np.max(epochs)), linewidth=1)
        ax1.scatter((X[-1],),(Y[-1],),s=10,c='k')
        Epoch.append(epch)
        Gain.append(conc_field.get_conc(X[-1],Y[-1])-conc_field.get_conc(X[0],Y[0]))
ax2.plot(Epoch,Gain)
ax2.set_xlabel('epoch')
ax2.set_ylabel(r'$\Delta c$')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
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
ax3.plot(coarseave(Epi,10),coarseave(Ret,10),'b-',label='return')
ax3.plot(coarseave(Epi,10),coarseave(Rew,10),'r-',label='accumulative reward')

#ax3.plot(Epi[9::10],Rew[9::10])
ax3.set_xlabel('epoch')
ax3.set_ylabel(r'reward')
ax3.legend(loc='best')
plt.show()
