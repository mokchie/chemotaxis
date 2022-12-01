import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import re,os
import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size':14, 'font.family':'sans-serif'})
cmap = cm.get_cmap('jet')
fig, (ax,ax1) = plt.subplots(1, 2)
pattern = re.compile("sample-epoch-([0-9]+).data$")
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
Gain  = []
Epoch = []
for epch, filename in files:
    X = []
    Y = []
    Xr = []
    Yr = []
    with open(direct+'/'+filename) as fp:
        for line in fp:
            t, x, y, tx, ty, nx, ny, kappa = [float(item) for item in line.strip().split()]
            #            if int(np.round(t/0.01))%20==0:
            #                ax.scatter([x,],[y,],c='r',s=5)
            X.append(x)
            Y.append(y)
            Xr.append(x+nx/kappa)
            Yr.append(y+ny/kappa)
    if epch % 1 == 0:
        X = np.array(X)
        Y = np.array(Y)
        x0 = 0
        y0 = 0
        X = X-x0
        Y = Y-y0
        Xr = np.array(Xr)-x0
        Yr = np.array(Yr)-y0
        ax1.plot(X, Y, '-', color=cmap(epch / np.max(epochs)), linewidth=1)
        ax1.plot(Xr, Yr, '.-', color=cmap(epch / np.max(epochs)), linewidth=1)
        ax1.scatter((X[-1],),(Y[-1],),s=10,c='k')
        Epoch.append(epch)
        Gain.append(Y[-1]-Y[0])
ax.plot(Epoch,Gain)
ax.set_xlabel('epoch')
ax.set_ylabel(r'$\Delta c$')
plt.show()