import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.rcParams.update({'font.size':14, 'font.family':'sans-serif'})
fig,(ax2,ax) = plt.subplots(2,1,sharex=True)
colors = ['C0','C1']

with open('./data/result.data','rb') as fp:
    for cn,state_size in enumerate([2,4]):
        Ep = np.load(fp)
        Ave = np.load(fp)
        Std = np.load(fp)
        ax.plot(Ep,Ave[:,0],color=colors[cn],label=r'$N_T=%s$'%state_size)
        ax.fill_between(Ep,Ave[:,0]-Std[:,0],Ave[:,0]+Std[:,0], color=colors[cn],alpha=0.2)
        ax2.plot(Ep,Ave[:,0],color=colors[cn],label=r'$N_T=%s$'%state_size)
        ax2.fill_between(Ep,Ave[:,0]-Std[:,0],Ave[:,0]+Std[:,0], color=colors[cn],alpha=0.2)
y0 = 162.10051016264254
ax.plot([Ep[0],Ep[-1]],[y0,y0],'k--')
ax.text(50,y0+1,r'$\Delta A_\mathrm{II}$')
ax2.plot([Ep[0],Ep[-1]],[y0,y0],'k--')
ax2.text(50,y0+1,r'$\Delta A_\mathrm{II}$')

y0 = 17.22748822645095
ax.plot([Ep[0],Ep[-1]],[y0,y0],'k--')
ax.text(50,y0-5,r'$\Delta A_\mathrm{IV}$')
ax2.plot([Ep[0],Ep[-1]],[y0,y0],'k--')
ax2.text(50,y0-5,r'$\Delta A_\mathrm{IV}$')
ax.set_ylim((-5,40))
ax2.set_ylim((150,180))
ax.set_xlabel('Epoch')
ax.set_ylabel('Error (degree)')
ax2.legend()

ax2.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.xaxis.tick_top()
ax2.tick_params(labeltop=False)  # don't put tick labels at the top
ax.xaxis.tick_bottom()

d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax2.transAxes, color='k', clip_on=False)
ax2.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax.transAxes)  # switch to the bottom axes
ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
ax.yaxis.set_label_coords(-0.1,1.1)

plt.show()
