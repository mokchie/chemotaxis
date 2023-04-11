import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import re,os,pdb
import numpy as np
matplotlib.rcParams.update({'font.size':14,'font.family':'sans-serif'})

tau_kappa_1 = np.linspace(-2,2,50)
tau_kappa_2 = np.linspace(-2,2,50)
tau_kappa_1,tau_kappa_2 = np.meshgrid(tau_kappa_1,tau_kappa_2)
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
    #return np.sqrt((np.sin(theta1)-np.sin(theta2))**2 + (np.cos(theta1)-np.cos(theta2))**2)
    return np.abs(theta1-theta2)/np.pi*180
deltah3 = Deltah3(tau_kappa_1,tau_kappa_2)
fig,ax = plt.subplots(1,1)
fig2,ax2 = plt.subplots(1,1)
CS = ax.contourf(tau_kappa_1,tau_kappa_2,deltah3,levels=40)
kappa1 = 5.5
kappa2 = 7.5
Taus = [(1,-1),
        (-5.7,-7.7),
        (7.7,5.7)]
annots = ['Case II', 'Case III','Case IV']
for i,(tau1,tau2) in enumerate(Taus):
    ax.annotate(annots[i],(tau1/kappa1,tau2/kappa2),xycoords='data',
                xytext=(-16, -30), textcoords='offset points',arrowprops=dict(edgecolor='C1', arrowstyle="->", connectionstyle="arc3"))
    ax2.annotate(annots[i],((tau1+tau2)/2,Deltah3(tau1/kappa1,tau2/kappa2)),xycoords='data',
                xytext=(-30, 30), textcoords='offset points',arrowprops=dict(edgecolor='C1', arrowstyle="->", connectionstyle="arc3"))    
    ax.plot((tau1/kappa1,),(tau2/kappa2,),'ro')
    ax2.plot((tau1+tau2)/2,Deltah3(tau1/kappa1,tau2/kappa2),'ro')
    print(Deltah3(tau1/kappa1,tau2/kappa2))
dtau = 2
xx = []
yy = []
Tau0 = np.linspace(-6.7,6.7,10)
for tau0 in Tau0:
    tau1=tau0+dtau/2
    tau2=tau0-dtau/2
    xx.append(tau1/kappa1)
    yy.append(tau2/kappa2)
xx = np.array(xx)
yy = np.array(yy)

#ax.plot(xx,yy,'C0')
ax.set_xlabel(r'$\tau_1/\kappa_1$')
ax.set_ylabel(r'$\tau_2/\kappa_2$')
cbar = fig.colorbar(CS)
cbar.ax.set_ylabel(r'$\Delta A$ (degree)')


ax2.plot(Tau0,Deltah3(xx,yy))
ax2.set_xlabel(r'$\bar{\tau}$')
ax2.set_ylabel(r'$|\Delta \mathbf{h}|$')
plt.show()

