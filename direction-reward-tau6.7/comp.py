import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import re,os
import numpy as np
matplotlib.rcParams.update({'font.size':14,'font.family':'sans-serif'})
state_size = 4
sname = 'test-DDQN-n%s'%(state_size,)
cmap = cm.get_cmap('jet')
ax1 = plt.figure().add_subplot(projection='3d')
fig2, ax2 = plt.subplots(1, 1)
fig3, ax3 = plt.subplots(1, 1)
pattern1 = re.compile(sname+"-swinging-epoch-([0-9]+).data$")
pattern2 = re.compile(sname+"-greedy-epoch-([0-9]+).data$")
pattern3 = re.compile(sname+"-DRL-epoch-([0-9]+).data$")
filenames1 = []
filenames2 = []
filenames3 = []
epochs1 = []
epochs2 = []
epochs3 = []
direct = "data"
for root, dirs, files in os.walk(direct):
    if root == direct:
        for name in files:
            found1 = pattern1.match(name)
            if found1:
                epochs1.append(int(found1.groups()[0]))
                filenames1.append(name)
                print(name)
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

R1 = []
R2 = []
R3 = []

files1 = sorted(zip(epochs1, filenames1))
files2 = sorted(zip(epochs2, filenames2))
files3 = sorted(zip(epochs3, filenames3))
files = zip(files1,files2,files3)
for (epch1, filename1), (epch2, filename2), (epch3, filename3) in files:
    Time1 = []
    X1 = []
    Y1 = []
    Z1 = []
    Xr1 = []
    Yr1 = []
    Zr1 = []
    Theta1 = []
    with open(direct+'/'+filename1) as fp1:
        for line in fp1:
            t, x, y, z, tx, ty, tz, nx, ny, nz, bx, by, bz, kappa, tau = [float(item) for item in line.strip().split()]
            #            if int(np.round(t/0.01))%20==0:
            #                ax.scatter([x,],[y,],c='r',s=5)
            r0 = kappa/(kappa**2+tau**2)
            Time1.append(t)
            X1.append(x)
            Y1.append(y)
            Z1.append(z)
            Xr1.append(x+nx*r0)
            Yr1.append(y+ny*r0)
            Zr1.append(z+nz*r0)
            phi = np.arctan(tau/kappa)
            if phi<0:
                phi+=np.pi
            helix_v = np.array([np.sin(phi)*tx,
                                np.sin(phi)*ty,
                                np.sin(phi)*tz]) \
                        + np.array([np.cos(phi) * bx,
                                    np.cos(phi) * by,
                                    np.cos(phi) * bz])
            theta = np.arccos(np.dot(helix_v, np.array([0,1,0])))/np.pi*180
            Theta1.append(theta)
            
    Time2 = []
    X2 = []
    Y2 = []
    Z2 = []
    Xr2 = []
    Yr2 = []
    Zr2 = []
    Theta2 = []
    with open(direct+'/'+filename2) as fp2:
        for line in fp2:
            t, x, y, z, tx, ty, tz, nx, ny, nz, bx, by, bz, kappa, tau = [float(item) for item in line.strip().split()]
            #            if int(np.round(t/0.01))%20==0:
            #                ax.scatter([x,],[y,],c='r',s=5)
            Time2.append(t)
            r0 = kappa/(kappa**2+tau**2)
            X2.append(x)
            Y2.append(y)
            Z2.append(z)
            Xr2.append(x+nx*r0)
            Yr2.append(y+ny*r0)
            Zr2.append(z+nz*r0)
            phi = np.arctan(tau/kappa)
            if phi<0:
                phi+=np.pi
            helix_v = np.array([np.sin(phi)*tx,
                                np.sin(phi)*ty,
                                np.sin(phi)*tz]) \
                        + np.array([np.cos(phi) * bx,
                                    np.cos(phi) * by,
                                    np.cos(phi) * bz])
            theta = np.arccos(np.dot(helix_v, np.array([0,1,0])))/np.pi*180
            Theta2.append(theta)

            
    X3 = []
    Y3 = []
    Z3 = []
    Xr3 = []
    Yr3 = []
    Zr3 = []
    Time3 = []
    Theta3 = []
    with open(direct+'/'+filename3) as fp3:
        for line in fp3:
            t, x, y, z, tx, ty, tz, nx, ny, nz, bx, by, bz, kappa, tau = [float(item) for item in line.strip().split()]
            #            if int(np.round(t/0.01))%20==0:
            #                ax.scatter([x,],[y,],c='r',s=5)
            r0 = kappa/(kappa**2+tau**2)
            Time3.append(t)
            X3.append(x)
            Y3.append(y)
            Z3.append(z)
            Xr3.append(x+nx*r0)
            Yr3.append(y+ny*r0)
            Zr3.append(z+nz*r0)
            phi = np.arctan(tau/kappa)
            if phi<0:
                phi+=np.pi
            helix_v = np.array([np.sin(phi)*tx,
                                np.sin(phi)*ty,
                                np.sin(phi)*tz]) \
                        + np.array([np.cos(phi) * bx,
                                    np.cos(phi) * by,
                                    np.cos(phi) * bz])
            theta = np.arccos(np.dot(helix_v, np.array([0,1,0])))/np.pi*180
            Theta3.append(theta)
    if epch1 % 1 == 0:

        X1 = np.array(X1)
        Y1 = np.array(Y1)
        Z1 = np.array(Z1)

        Xr1 = np.array(Xr1)
        Yr1 = np.array(Yr1)
        Zr1 = np.array(Zr1)

        X2 = np.array(X2)
        Y2 = np.array(Y2)
        Z2 = np.array(Z2)

        Xr2 = np.array(Xr2)
        Yr2 = np.array(Yr2)
        Zr2 = np.array(Zr2)

        Time3 = np.array(Time3)
        X3 = np.array(X3)
        Y3 = np.array(Y3)
        Z3 = np.array(Z3)

        Xr3 = np.array(Xr3)
        Yr3 = np.array(Yr3)
        Zr3 = np.array(Zr3)
        Theta3 = np.array(Theta3)
        if epch1%4==0:
            ax1.plot3D(X1, Y1, Z1, '--', color=cmap(epch1 / np.max(epochs1)), linewidth=1)
            #ax2.plot3D(Xr1, Yr1, Zr1, '--', color=cmap(epch1 / np.max(epochs1)), linewidth=1)
            ax1.scatter3D((X1[-1],),(Y1[-1],),(Z1[-1]),s=10,c='k')
            #ax2.plot(Time1,Theta1,'--', color=cmap(epch1 / np.max(epochs1)), linewidth=1)            

            ax1.plot3D(X2, Y2, Z2, '-.', color=cmap(epch2 / np.max(epochs2)), linewidth=1)
            #ax2.plot3D(Xr2, Yr2, Zr2, '-.', color=cmap(epch2 / np.max(epochs2)), linewidth=1)
            ax1.scatter3D((X2[-1],),(Y2[-1],),(Z2[-1]),s=10,c='k')
            #ax2.plot(Time2,Theta2,'-.', color=cmap(epch2 / np.max(epochs2)), linewidth=1)            

            ax1.plot3D(X3, Y3, Z3,'-', color=cmap(epch3 / np.max(epochs3)), linewidth=1)
            #ax2.plot3D(Xr3, Yr3, Zr3, '-', color=cmap(epch3 / np.max(epochs3)), linewidth=1)
            ax1.scatter3D((X3[-1],),(Y3[-1],),(Z3[-1]),s=10,c='k')
            ax2.plot(Time3,Theta3,'-', color=cmap(epch3 / np.max(epochs3)), linewidth=1,alpha=0.5)

        R1.append(Y1[-1]-Y1[0])
        R2.append(Y2[-1]-Y2[0])
        R3.append(Y3[-1]-Y3[0])
ax3.plot(np.array(range(len(R1)))+1,R1,'o--', color='C0',label='alternating')
ax3.plot(np.array(range(len(R1)))+1,np.average(R1)+np.zeros_like(R1),'-',color='C0')
ax3.plot(np.array(range(len(R2)))+1,R2,'v--',color='C1',label='short-sighted')
ax3.plot(np.array(range(len(R2)))+1,np.average(R2)+np.zeros_like(R2),'-',color='C1')
ax3.plot(np.array(range(len(R3)))+1,R3,'s--',color='C2',label='DRL')
ax3.plot(np.array(range(len(R3)))+1,np.average(R3)+np.zeros_like(R3),'-',color='C2')
y0 = 17.2
ax2.plot(Time3,np.zeros_like(Time3)+y0,'k--')
ax2.text(40,y0+2,r'$\Delta A_\mathrm{IV}$')

# ax.scatter([0,],[0,],s=10,c='r')
#ax1.set_aspect('equal')
#ax1.set_xlim((-10,10))
ax1.set_xlabel('x')
ax1.set_ylabel('y')
#ax2.set_aspect('equal')
#ax2.set_xlim((-35,25))
#ax2.set_xlabel('x')
#ax2.set_ylabel('y')
ax2.set_xlabel('t')
ax2.set_ylabel(r'$\theta_h$ (degree)')
ax3.set_xlabel(r'$i$')
ax3.set_ylabel(r'$\Delta c/k_c$')
#ax3.set_ylim((10,30))
ax3.legend(loc='upper left',ncol=2)
ax3.set_ylim((-100,200))
plt.show()
