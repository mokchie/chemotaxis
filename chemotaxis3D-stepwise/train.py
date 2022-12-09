from swimmer import *
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sname = 'sample'
clear(sname)
random.seed(39895)
conc_field = Conc_field(c0=20,k=1)
state_size = 8
swimmer = Swimmer(v0=1,
                  k0=6.5, kw=2, kn=5,
                  tau0=6.7, tauw=2, taun=5,
                  t0=0,
                  rx0=2, ry0=10, rz0=4,
                  tx0=1, ty0=0, tz0=0,
                  nx0=0, ny0=-1, nz0=0,
                  Taction=1/state_size,
                  dt=0.008,
                  field=conc_field,
                  targetx=0, targety=2000, targetz=0,
                  lifespan=160,
                  state_size=state_size,
                  Regg=0.5,
                  sname=sname,
                  xb=[0,10], yb=[0,10],zb=[0,10],
                  rand=True,
                  dump_freq=4,
                  dim=3)

agent = DQN(swimmer, epochs=1600, batch_size=128, gamma=0.98, epsilon_min=0.1, epsilon_decay=0.998)

scores = agent.train()
agent.model.save('saved_model/saved_model1')

plt.plot(scores)

plt.show()