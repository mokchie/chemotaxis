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
fig2, (ax21,ax22) = plt.subplots(2, 1,sharex=True)
fig3, (ax31,ax32,ax33) = plt.subplots(3, 1,sharex=True)
N = 4
filenames1 = [sname+"-swinging-epoch-%s.data"%N,]
filenames2 = [sname+"-greedy-epoch-%s.data"%N,]
filenames3 = [sname+"-DRL-epoch-%s.data"%N,]
epochs1 = [N,]
epochs2 = [N,]
epochs3 = [N,]
direct = "data"
R1 = []
R2 = []
R3 = []
files = sorted(zip(epochs1, filenames1, epochs2, filenames2, epochs3, filenames3))
tstop = 10
for epch1, filename1, epch2, filename2, epch3, filename3 in files:
    X1 = []
    Y1 = []
    Z1 = []
    Xr1 = []
    Yr1 = []
    Zr1 = []
    Time1 = []
    Kappa1 = []
    Theta1 = []
    with open(direct+'/'+filename1) as fp1:
        for line in fp1:
            t, x, y, z, tx, ty, tz, nx, ny, nz, bx, by, bz, kappa, tau = [float(item) for item in line.strip().split()]
            #            if int(np.round(t/0.01))%20==0:
            #                ax.scatter([x,],[y,],c='r',s=5)
            if t>tstop:
                break
            Time1.append(t)
            Kappa1.append(kappa)
            X1.append(x)
            Y1.append(y)
            Z1.append(z)
            Xr1.append(x+nx/kappa)
            Yr1.append(y+ny/kappa)
            Zr1.append(z+nz/kappa)
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
    Kappa2 = []
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
            if t>tstop:
                break            
            X2.append(x)
            Y2.append(y)
            Z2.append(z)
            Xr2.append(x+nx/kappa)
            Yr2.append(y+ny/kappa)
            Zr2.append(z+nz/kappa)
            Time2.append(t)
            Kappa2.append(kappa)

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
            
    Time3 = []
    Kappa3 = []
    X3 = []
    Y3 = []
    Z3 = []
    Xr3 = []
    Yr3 = []
    Zr3 = []
    Theta3 = []
    with open(direct+'/'+filename3) as fp3:
        for line in fp3:
            t, x, y, z, tx, ty, tz, nx, ny, nz, bx, by, bz, kappa, tau = [float(item) for item in line.strip().split()]
            #            if int(np.round(t/0.01))%20==0:
            #                ax.scatter([x,],[y,],c='r',s=5)
            if t>tstop:
                break            
            Time3.append(t)
            Kappa3.append(kappa)
            X3.append(x)
            Y3.append(y)
            Z3.append(z)
            Xr3.append(x+nx/kappa)
            Yr3.append(y+ny/kappa)
            Zr3.append(z+nz/kappa)
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

        X3 = np.array(X3)
        Y3 = np.array(Y3)
        Z3 = np.array(Z3)

        Xr3 = np.array(Xr3)
        Yr3 = np.array(Yr3)
        Zr3 = np.array(Zr3)

        Time1 = np.array(Time1)
        Kappa1 = np.array(Kappa1)
        Time2 = np.array(Time2)
        Kappa2 = np.array(Kappa2)
        Time3 = np.array(Time3)
        Kappa3 = np.array(Kappa3)        

        if epch1%1==0:
            ax1.plot3D(X1, Y1, Z1,'-', color='C0', linewidth=1,label='alternating')
            ax21.plot(Time1,Y1-Y1[0], '-', color='C0', linewidth=1,label='alternating')
            ax22.plot(Time1,Theta1, '-', color='C0', linewidth=1,label='alternating')            
            ax1.scatter3D((X1[0],), (Y1[0],), (Z1[0],),s=10, c='r', zorder=10)
            ax1.scatter((X1[-1],), (Y1[-1],), (Z1[-1],), s=10, c='k',zorder=10)
            ax31.plot(Time1,Kappa1,'-',color='C0',linewidth=1)

            ax1.plot3D(X2, Y2, Z2, '-', color='C1', linewidth=1,label='short-sighted')
            ax21.plot(Time2,Y2-Y2[0], '-', color='C1', linewidth=1,label='short-sighted')
            ax22.plot(Time2,Theta2, '-', color='C1', linewidth=1,label='short-sighted')                        
            ax1.scatter3D((X2[-1],),(Y2[-1],),(Z2[-1],),s=10,c='k',zorder=10)
            ax32.plot(Time2, Kappa2, '-', color='C1', linewidth=1)

            ax1.plot3D(X3, Y3, Z3, '-', color='C2', linewidth=1,label='DRL')
            ax21.plot(Time3,Y3-Y3[0], '-', color='C2', linewidth=1,label='DRL')
            ax22.plot(Time3,Theta3, '-', color='C2', linewidth=1,label='DRL')            
            ax1.scatter3D((X3[-1],),(Y3[-1],),(Z3[-1],),s=10,c='k',zorder=10)
            ax33.plot(Time3, Kappa3, '-', color='C2', linewidth=1)

        R1.append(Y1[-1]-Y1[0])
        R2.append(Y2[-1]-Y2[0])
        R3.append(Y3[-1]-Y3[0])
# ax.scatter([0,],[0,],s=10,c='r')
#ax1.set_aspect('equal')
#ax1.set_xlim((-10,10))
for X,Y,Z in zip((X1,X2,X3),(Y1,Y2,Y3),(Z1,Z2,Z3)):
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
       ax1.plot([xb], [yb], [zb], 'w')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax22.set_xlabel(r'$t$')
ax21.set_ylabel(r'$\Delta c/k_c$')
ax21.set_ylim((-1,13))
ax22.set_ylabel(r'$\theta_h$ (degree)')
ax21.legend(loc='best',ncol=2)
#ax22.legend(loc='best',ncol=2)
ax33.set_xlabel(r'$t$')
ax31.set_ylabel(r'$\kappa$')
ax32.set_ylabel(r'$\kappa$')
ax33.set_ylabel(r'$\kappa$')
ax1.legend(loc='best',ncol=2)
#ax1.set_xlim((27,52))
#ax1.set_ylim((40,58))
#axin.set_xlim((41,44))
#axin.set_ylim((44,47))
#ax31.set_xlim((0,80))
fig3.set_size_inches(10, 2)
#ax3.set_ylim((10,30))
plt.show()
