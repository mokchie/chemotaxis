import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import re,os
import numpy as np
import pdb
matplotlib.rcParams.update({'font.size':14,'font.family':'sans-serif'})
state_size = 4
cmap = cm.get_cmap('jet')
fig1, ax1 = plt.subplots(1, 1)
RR = []
Tau0 = np.linspace(-6.7,6.7,10)
for tau0 in Tau0:
    sname = 'test-DDQN-n%s-tau%.2f'%(state_size,tau0)
    pattern2 = re.compile(sname+"-greedy-epoch-([0-9]+).data$")
    filenames2 = []
    epochs2 = []
    direct = "data"
    for root, dirs, files in os.walk(direct):
        if root == direct:
            for name in files:
                found2 = pattern2.match(name)
                if found2:
                    epochs2.append(int(found2.groups()[0]))
                    filenames2.append(name)
                    print(name)
    R2 = []
    files = sorted(zip(epochs2, filenames2))
    for epch2, filename2 in files:
        X2 = []
        Y2 = []
        Z2 = []
        Xr2 = []
        Yr2 = []
        Zr2 = []
        with open(direct+'/'+filename2) as fp2:
            for line in fp2:
                t, x, y, z, tx, ty, tz, nx, ny, nz, bx, by, bz, kappa, tau = [float(item) for item in line.strip().split()]
                #            if int(np.round(t/0.01))%20==0:
                #                ax.scatter([x,],[y,],c='r',s=5)
                X2.append(x)
                Y2.append(y)
                Z2.append(z)
                Xr2.append(x+nx/kappa)
                Yr2.append(y+ny/kappa)
                Zr2.append(z+nz/kappa)
        X2 = np.array(X2)
        Y2 = np.array(Y2)
        Z2 = np.array(Z2)
        Xr2 = np.array(Xr2)
        Yr2 = np.array(Yr2)
        Zr2 = np.array(Zr2)
        R2.append(Y2[-1]-Y2[0])
    RR.append(R2)
RR = np.array(RR)
def Deltah3(tau_kappa_1,tau_kappa_2):
    theta1 = np.arctan(tau_kappa_1)
    theta2 = np.arctan(tau_kappa_2)
    if isinstance(theta1,np.ndarray):
        theta1_c = []
        for th in theta1.flatten():
            if th<0:
                theta1_c.append(th+np.pi)
            else:
                theta1_c.append(th)
        theta1 = np.array(theta1_c).reshape(theta1.shape)
    else:
        if theta1<0:
            theta1+=np.pi
    if isinstance(theta2,np.ndarray):
        theta2_c = []
        for th in theta2.flatten():
            if th<0:
                theta2_c.append(th+np.pi)
            else:
                theta2_c.append(th)
        theta2 = np.array(theta2_c).reshape(theta2.shape)
    else:
        if theta2<0:
            theta2+=np.pi
    return np.sqrt((np.sin(theta1)-np.sin(theta2))**2 + (np.cos(theta1)-np.cos(theta2))**2)
kappa1 = 5.5
kappa2 = 7.5
dtau = 2
xx = []
yy = []
for tau0 in Tau0:
    tau1=tau0+dtau/2
    tau2=tau0-dtau/2
    xx.append(tau1/kappa1)
    yy.append(tau2/kappa2)
xx = np.array(xx)
yy = np.array(yy)

deltah3 = Deltah3(xx,yy)
#ax1.errorbar(deltah3,np.average(RR,axis=1),yerr=np.std(RR,axis=1),marker='o')
ax1.errorbar(Tau0,np.average(RR,axis=1),yerr=np.std(RR,axis=1),marker='o')
ax1.set_xlabel(r'$|\Delta h_3|$')
ax1.set_ylabel(r'$\Delta c/k_c$')
#ax1.legend(loc='upper left',ncol=2)

plt.show()
