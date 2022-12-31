import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import re,os
import numpy as np
matplotlib.rcParams.update({'font.size':14,'font.family':'sans-serif'})
state_size = 2
epsilon = 0.2
sname = 'test-n%s-epsilon%s'%(state_size,epsilon)
cmap = cm.get_cmap('jet')
ax1 = plt.figure().add_subplot(projection='3d')
fig2, ax2 = plt.subplots(1, 1)
#pattern1 = re.compile(sname+"-swinging-epoch-([0-9]+).data$")
pattern2 = re.compile(sname+"-greedy-epoch-([0-9]+).data$")
pattern3 = re.compile(sname+"-DRL-epoch-([0-9]+).data$")
#filenames1 = []
filenames2 = []
filenames3 = []
#epochs1 = []
epochs2 = []
epochs3 = []
direct = "data"
for root, dirs, files in os.walk(direct):
    if root == direct:
        for name in files:
            # found1 = pattern1.match(name)
            # if found1:
            #     epochs1.append(int(found1.groups()[0]))
            #     filenames1.append(name)
            #     print(name)
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

#R1 = []
R2 = []
R3 = []
files = sorted(zip(epochs2, filenames2, epochs3, filenames3))
for epch2, filename2, epch3, filename3 in files:
    # X1 = []
    # Y1 = []
    # Z1 = []
    # with open(direct+'/'+filename1) as fp1:
    #     for line in fp1:
    #         t, x, y, z, tx, ty, tz, nx, ny, nz, bx, by, bz, kappa, tau = [float(item) for item in line.strip().split()]
    #         #            if int(np.round(t/0.01))%20==0:
    #         #                ax.scatter([x,],[y,],c='r',s=5)
    #         X1.append(x)
    #         Y1.append(y)
    #         Z1.append(z)
    X2 = []
    Y2 = []
    Z2 = []
    with open(direct+'/'+filename2) as fp2:
        for line in fp2:
            t, x, y, z, tx, ty, tz, nx, ny, nz, bx, by, bz, kappa, tau = [float(item) for item in line.strip().split()]
            #            if int(np.round(t/0.01))%20==0:
            #                ax.scatter([x,],[y,],c='r',s=5)
            X2.append(x)
            Y2.append(y)
            Z2.append(z)

    X3 = []
    Y3 = []
    Z3 = []

    with open(direct+'/'+filename3) as fp3:
        for line in fp3:
            t, x, y, z, tx, ty, tz, nx, ny, nz, bx, by, bz, kappa, tau = [float(item) for item in line.strip().split()]
            #            if int(np.round(t/0.01))%20==0:
            #                ax.scatter([x,],[y,],c='r',s=5)
            X3.append(x)
            Y3.append(y)
            Z3.append(z)
    if epch2 % 1 == 0:
        # X1 = np.array(X1)
        # Y1 = np.array(Y1)
        # Z1 = np.array(Z1)
        X2 = np.array(X2)
        Y2 = np.array(Y2)
        Z2 = np.array(Z2)
        X3 = np.array(X3)
        Y3 = np.array(Y3)
        Z3 = np.array(Z3)

        if epch2%4==0:
            #ax1.plot(X1, Y1, '--', color=cmap(epch1 / np.max(epochs1)), linewidth=1)
            #ax2.plot(Xr1, Yr1, '--', color=cmap(epch1 / np.max(epochs1)), linewidth=1)
            #ax1.scatter((X1[-1],),(Y1[-1],),s=10,c='k')
            #ax1.plot3D(X1, Y1, Z1, '--', color=cmap(epch1 / np.max(epochs1)), linewidth=1)
            #ax1.scatter((X2[-1],),(Y2[-1],),(Z2[-1],), s=10,c='k')

            ax1.plot3D(X2, Y2, Z2, '-.', color=cmap(epch2 / np.max(epochs2)), linewidth=1)
            ax1.scatter((X2[-1],),(Y2[-1],),(Z2[-1],), s=10,c='k')

            ax1.plot(X3, Y3, Z3, '-', color=cmap(epch3 / np.max(epochs3)), linewidth=1)
            ax1.scatter((X3[-1],),(Y3[-1],),(Z3[-1],),s=10,c='k')

        # R1.append(Y1[-1]-Y1[0])
        R2.append(Y2[-1]-Y2[0])
        R3.append(Y3[-1] - Y3[0])
#ax2.plot(np.array(range(len(R1)))+1,R1,'o--', color='C1',label='swinging')
#ax2.plot(np.array(range(len(R1)))+1,np.average(R1)+np.zeros_like(R1),'-',color='C1')
ax2.plot(np.array(range(len(R2)))+1,R2,'v--',color='C2',label='greedy')
ax2.plot(np.array(range(len(R2)))+1,np.average(R2)+np.zeros_like(R2),'-',color='C2')
ax2.plot(np.array(range(len(R3)))+1,R3,'v--',color='C3',label='DRL')
ax2.plot(np.array(range(len(R3)))+1,np.average(R3)+np.zeros_like(R3),'-',color='C3')
# ax.scatter([0,],[0,],s=10,c='r')
ax1.set_aspect('auto')
#ax1.set_xlim((-10,10))
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')

ax2.set_xlabel(r'$i$')
ax2.set_ylabel(r'$\Delta c$')
#ax3.set_ylim((10,30))
ax2.legend(loc='best')
plt.show()