import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import re,os
import numpy as np
matplotlib.rcParams.update({'font.size':14,'font.family':'sans-serif'})
cmap = cm.get_cmap('jet')
fig2, ax2 = plt.subplots(1, 1)
direct = "data"
colors1 = ['C0','C1']
colors2 = ['C2','C3']
for cn,state_size in enumerate([2, 4]):
    XI = np.array([0.0,0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.2])
    DeltaC = []
    DeltaCErr = []
    for xi in XI:
        sname = 'test-n%s-xi%s'%(state_size,xi)
        #pattern1 = re.compile(sname+"-swinging-epoch-([0-9]+).data$")
        pattern2 = re.compile(sname+"-greedy-epoch-([0-9]+).data$")
        pattern3 = re.compile(sname+"-DRL-epoch-([0-9]+).data$")
        #filenames1 = []
        filenames2 = []
        filenames3 = []
        #epochs1 = []
        epochs2 = []
        epochs3 = []
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
            X2 = []
            Y2 = []
            with open(direct+'/'+filename2) as fp2:
                for line in fp2:
                    t, x, y, tx, ty, nx, ny, kappa = [float(item) for item in line.strip().split()]
                    #            if int(np.round(t/0.01))%20==0:
                    #                ax.scatter([x,],[y,],c='r',s=5)
                    X2.append(x)
                    Y2.append(y)

            X3 = []
            Y3 = []
            with open(direct+'/'+filename3) as fp3:
                for line in fp3:
                    t, x, y, tx, ty, nx, ny, kappa = [float(item) for item in line.strip().split()]
                    #            if int(np.round(t/0.01))%20==0:
                    #                ax.scatter([x,],[y,],c='r',s=5)
                    X3.append(x)
                    Y3.append(y)
            if epch2 % 1 == 0:

                X2 = np.array(X2)
                Y2 = np.array(Y2)
                X3 = np.array(X3)
                Y3 = np.array(Y3)
                # R1.append(Y1[-1]-Y1[0])
                R2.append(Y2[-1]-Y2[0])
                R3.append(Y3[-1] - Y3[0])
        #ax2.plot(np.array(range(len(R1)))+1,R1,'o--', color='C1',label='swinging')
        #ax2.plot(np.array(range(len(R1)))+1,np.average(R1)+np.zeros_like(R1),'-',color='C1')
        #ax2.plot(np.array(range(len(R2)))+1,np.average(R2)+np.zeros_like(R2),'-',color='C2')
        DeltaC.append([np.average(R2),np.average(R3)])
        DeltaCErr.append([np.std(R2),np.std(R3)])
        #ax2.plot(np.array(range(len(R3)))+1,np.average(R3)+np.zeros_like(R3),'-',color='C3')
# ax.scatter([0,],[0,],s=10,c='r')
    DeltaC = np.array(DeltaC)
    DeltaCErr = np.array(DeltaCErr)
    width = 0.003
    error_params = dict(elinewidth=1,ecolor='k',capsize=1)
    Xe = XI
    ax2.bar(Xe-width/2+(cn*2-1)*width,DeltaC[:,0],width=width,yerr=DeltaCErr[:,0],error_kw=error_params,color=colors1[cn],label='$greedy, N_T=%s$'%state_size)
    ax2.bar(Xe+width/2+(cn*2-1)*width, DeltaC[:,1],width=width,yerr=DeltaCErr[:,1], error_kw = error_params, color=colors2[cn],label='$DRL, N_T=%s$'%state_size)
    tick_label = [r'$%s$'%xi for xi in XI]
    plt.xticks(Xe,tick_label)
#ax2.set_xlabel(r'$i$')
ax2.set_ylabel(r'$\Delta c/c_k$')
ax2.set_xlabel(r'$\xi$')
#ax3.set_ylim((10,30))
ax2.legend(loc='best',ncol=2)
plt.show()