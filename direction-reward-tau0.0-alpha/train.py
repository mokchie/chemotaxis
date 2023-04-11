import time
import os,sys
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)
del path
from swimmer_v2 import *
#import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK']='True'
conc_field = Conc_field(c0=20,k=1)
for state_size in [2,4]:
    sname = 'sample-DDQN-n%s'%state_size
    clear(sname)
    random.seed(39895)
    swimmer = Swimmer(dim=3,
                      v0=2,
                      vw=0.2,
                      k0=6.5, kw=2.0, kn=2,
                      tau0=0.0, tauw=2.0, taun=2,
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
                      dump_freq=25,
                      saving_interval_dt=10,
                      actionAll=False,
                      accurate_reward=True,
                      direction_reward=True)

    agent = DQN(swimmer,
                epochs=1600,
                batch_size=128,
                gamma=0.1,
                epsilon_min=0.1,
                epsilon_decay=0.998,
                N_neurons=32,
                N_hidden=4,
                DDQN=True,
                align_freq=25,
                alpha = 0.002
                )

    scores = agent.train()
    agent.Q_network.save('saved_model/saved_model_'+sname)

    #plt.plot(scores)

#plt.show()
