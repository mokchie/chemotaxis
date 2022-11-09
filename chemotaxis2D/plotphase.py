import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from swimmer import Conc_field
conc_field = Conc_field(c0=20,k=1)
matplotlib.rcParams.update({'font.size':14,'font.family':'sans-serif'})
cmap = cm.get_cmap('viridis')
fig0, ax0 = plt.subplots(1, )
fig1, ax1 = plt.subplots(1, )
n = 10
state_size = 2
sname = 'test-%s'%state_size

filename = sname+'-2-epoch-%s.data'%n
labels = ['DRL']
colors = ['C0','C1','C2']
direct = "data"
k0 = 4.0
Taction = 1/state_size
v0 = 1
dt = 0.02
ntimes = int(Taction * 2 * np.pi / k0 / v0 / dt)

X = []
Y = []
Xr = []
Yr = []
Kappa = []
Time = []
C = []
Ns = 0
Nx = []
Ny = []
jj = 0
with open(direct+'/'+filename) as fp:
    for line in fp:
        t, x, y, nx, ny, kappa = [float(item) for item in line.strip().split()]
        X.append(x)
        Y.append(y)
        C.append(conc_field.get_conc(x,y))
        xr = x+nx/kappa
        Xr.append(xr)
        yr = y+ny/kappa
        Yr.append(yr)
        Time.append(t)
        if jj%ntimes==0:
            Kappa.append(kappa)
            Nx.append(nx)
            Ny.append(ny)
        x0 = x
        y0 = y
        Ns += 1
        jj += 1
ax0.plot(X, Y, '-', color=colors[0], label=labels[0], linewidth=1)
k1 = np.min(Kappa)
k2 = np.max(Kappa)
Nb = len(Kappa)
print(Nb)
shrink = 0.2
theta = np.linspace(0,2*np.pi,200)
#ax1.plot(np.cos(theta),np.sin(theta),'k-')
for j,(nx,ny,kappa) in enumerate(zip(Nx,Ny,Kappa)):
    if j in list(range(8)):
        continue
    if j>Nb:
        break
    symsize = 40
    if kappa==k1:
        marker = 'o'
    else:
        marker = '*'
    #ax1.scatter([nx/kappa,],[ny/kappa,],s=symsize,color=cmap(j/Nb),marker=marker)
    ax1.scatter([nx*(1-j/Nb*shrink),],[ny*(1-j/Nb*shrink),],s=symsize,color=cmap(j/Nb),marker=marker)
# ax.scatter([0,],[0,],s=10,c='r')
ax0.set_aspect('equal')
ax1.set_aspect('equal')
ax0.set_xlabel('$x$')
ax0.set_ylabel('$y$')
ax0.legend(loc = 'upper right')
ax0.set_xlim((0,25))
#ax1.set_xlim((-1.1,1.1))
#ax1.set_ylim((-1.1,1.1))
ax1.set_xticks([])
ax1.set_yticks([])
plt.show()
