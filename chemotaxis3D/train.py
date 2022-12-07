from swimmer import *
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sname = 'sample'
clear(sname)
random.seed(39895)
conc_field = Conc_field(c0=20,k=1)
state_size = 4
swimmer = Swimmer(v0=1, k0=4.0, kw=2.0, kn=2, t0=0, rx0=2, ry0=10, tx0=1, ty0=0, nx0=0, ny0=-1, Taction=1/state_size, dt=0.02, field=conc_field,targetx=0, targety=1000, lifespan=160, state_size=state_size, sname=sname, xb=[0,10], yb=[0,10],zb=[0,10],rand=True, dump_freq=10, tau0=1.0, tauw=0.2, taun=16, dim=3)

agent = DQN(swimmer, epochs=1600, batch_size=256, gamma=0.98, epsilon_min=0.1, epsilon_decay=0.998)

scores = agent.train()
agent.model.save('saved_model/saved_model1')

plt.plot(scores)

plt.show()