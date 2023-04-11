import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import re,os
import numpy as np
matplotlib.rcParams.update({'font.size':14,'font.family':'sans-serif'})

kappa1 = 5.5
kappa2 = 7.5

def r0(kappa,tau):
    return kappa/(kappa**2+tau**2)
def h0(kappa,tau):
    return tau/(kappa**2+tau**2)
def theta0(kappa,tau):
    return np.arctan(h0(kappa,tau)/r0(kappa,tau))
def h0_r0(kappa,tau):
    return kappa*tau/(kappa**2+tau**2)

Tau00 = np.linspace(-6.7,6.7,50)
Dtau0 = np.linspace(0,4,50)
Tau0,Dtau = np.meshgrid(Tau00,Dtau0)

Tau1 = Tau0+Dtau/2
Tau2 = Tau0-Dtau/2

Delta_h3 = np.sqrt((np.sin(theta0(kappa1,Tau1))-np.sin(theta0(kappa2,Tau2)))**2 + (np.cos(theta0(kappa1,Tau1))-np.cos(theta0(kappa2,Tau2)))**2)
fig,ax = plt.subplots(1,1)
CS = ax.contourf(Tau0,Dtau,Delta_h3,levels=40)
ax.set_xlabel(r'$\bar{\tau}$')
ax.set_ylabel(r'$\Delta \tau$')
cbar = fig.colorbar(CS)
cbar.ax.set_ylabel(r'$\Delta \mathbf{h}_3$')

kappa1 = 5.5
kappa2 = 7.5
Taus = [(1,-1),
        (-5.7,-7.7),
        (7.7,5.7)]
annots = ['Case 2', 'Case 3','Case 4']
for i,(tau1,tau2) in enumerate(Taus):
    ax.annotate(annots[i],((tau1+tau2)/2,tau1-tau2),xycoords='data',
                xytext=(-16, -30), textcoords='offset points',arrowprops=dict(edgecolor='C1', arrowstyle="->", connectionstyle="arc3"))


plt.show()

