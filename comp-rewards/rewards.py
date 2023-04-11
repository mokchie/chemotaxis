import os,sys,pdb,random
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)
del path
from swimmer_v2 import *
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.rcParams.update({'font.size':14, 'font.family':'sans-serif'})
conc_field = Conc_field(c0=20,k=1)
fig1,ax1 = plt.subplots(1,1)
fig2,ax2 = plt.subplots(1,1)
state_size = 2
sname = 'comp-rewards-NN-n%s'%state_size
random.seed(39895)
kappa0 = 6.5
tau0 = 6.7
swimmer = Swimmer(dim=3,
                  v0=2,
                  vw=0.2,
                  k0=kappa0, kw=2, kn=2,
                  tau0=tau0, tauw=2, taun=2,
                  t0=0,
                  rx0=2, ry0=10, rz0=0,
                  tx0=1, ty0=0, tz0=0,
                  nx0=0, ny0=-1, nz0=0,
                  Taction=1/state_size,
                  dt=0.002,
                  conc_field=conc_field,
                  targetx=0, targety=1000, targetz=0,
                  lifespan=80,
                  state_size=state_size,
                  sname=sname,
                  xb=[0,10],yb=[0,10],zb=[0,10],
                  rand=True,
                  dump_freq=1000,
                  saving_interval_dt=10,
                  actionAll=False)
tot = 10000
data_file = 'data/comp.data'
if os.path.isfile(data_file) and len(sys.argv)==1:
    with  open(data_file,'rb') as fp:
        Deltac = np.load(fp)
        Deltah = np.load(fp)
else:
    Deltac = []
    Deltah = []
    for i in range(tot):
        print('%s/%s'%(i,tot))
        states = swimmer.reset()
        kappa1,tau1 = swimmer.actions[0]
        kappa2,tau2 = swimmer.actions[1]
        swimmer.kappa = kappa1
        swimmer.tau = tau1
        rc1 = swimmer.get_center()
        c1 =  conc_field.get_conc(rc1[0],rc1[1],rc1[2])        
        h1 = swimmer.get_h()
        swimmer.kappa = kappa2
        swimmer.tau = tau2
        rc2 = swimmer.get_center()
        c2 =  conc_field.get_conc(rc2[0],rc2[1],rc2[2])        
        h2 = swimmer.get_h()
        
        deltac = c2-c1
        deltah = np.dot(h2,np.array([0,1,0])) - np.dot(h1,np.array([0,1,0]))
        Deltac.append(deltac)
        Deltah.append(deltah)
        Deltac.append(-deltac)
        Deltah.append(-deltah)        
    with  open(data_file,'wb') as fp:
        np.save(fp,np.array(Deltac))
        np.save(fp,np.array(Deltah))
Deltac = np.array(Deltac)*10
xx_min = np.min(Deltac)
xx_max = np.max(Deltac)
dxx = xx_max-xx_min
xx_min -= dxx*0.1
xx_max += dxx*0.1
xx = np.linspace(xx_min,xx_max,20)
yy_min = np.min(Deltah)
yy_max = np.max(Deltah)
dyy = yy_max-yy_min
yy_min -= dyy*0.1
yy_max += dyy*0.1
yy = np.linspace(yy_min,yy_max,20)

xxc = (xx[0:-1]+xx[1:])/2
yyc = (yy[0:-1]+yy[1:])/2
dx = xx[1]-xx[0]
dy = yy[1]-yy[0]

X,Y = np.meshgrid(xxc,yyc,indexing='ij')
Z = np.zeros_like(X)

for c,h in zip(Deltac,Deltah):
    i = int((c-xx[0])/dx)
    j = int((h-yy[0])/dy)
    Z[i,j]+=1
Z /= np.sum(Z)
ax1.plot(Deltac,Deltah,'.')
ax2.contourf(X,Y,Z,levels=10)
ax1.set_xlabel(r'$\Delta c$')
ax1.set_ylabel(r'$\Delta (\mathbf{h}\cdot \mathbf{e}_y)$')
ax2.set_xlabel(r'$\Delta c$')
ax2.set_ylabel(r'$\Delta (\mathbf{h}\cdot \mathbf{e}_y)$')
ax1.set_aspect('equal')
ax2.set_aspect('equal')
ax1.set_xlim((-0.35,0.35))
ax1.set_ylim((-0.35,0.35))
ax2.set_xlim((-0.35,0.35))
ax2.set_ylim((-0.35,0.35))
plt.show()
