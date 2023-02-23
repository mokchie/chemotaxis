import os,sys
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)
del path
from swimmer_v2 import *
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK']='True'
state_size = 4
sname = 'sample-n%s'%state_size
clear(sname)
random.seed(39895)
conc_field = Conc_field(c0=20,k=1)

swimmer = Swimmer(dim=3,
                  v0=2,
                  k0=6.5, kw=2.0, kn=2,
                  tau0=6.7, tauw=2.0, taun=2,
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
                  dump_freq=50,
                  actionAll=False,
                  saving_interval_dt=10,
                  sigma_kappa=0.1,
                  sigma_tau=0.1)

agent = DQN(swimmer,
            epochs=1600,
            batch_size=128,
            gamma=0.9,
            epsilon_min=0.1,
            epsilon_decay=0.998,
            N_neurons=32,
            N_hidden=4,
            )

scores = agent.train()
agent.model.save('saved_model/saved_model_'+sname)

plt.plot(scores)

plt.show()