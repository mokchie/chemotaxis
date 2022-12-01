from swimmer import *
import sys,os,random
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
random.seed(39895)
state_size = 8
sname = 'sample-%s'%state_size
clear(sname)
conc_field = Conc_field(c0=20,k=1)
swimmer = Swimmer(v0=1, vw=0.2, k0=4.0, kw=2.0, kn=2, t0=0, rx0=2, ry0=10, tx0=1, ty0=0, nx0=0, ny0=-1, Taction=1/state_size, dt=0.01, field=conc_field.get_conc,targetx=0, targety=1000, lifespan=80, state_size=state_size, sname=sname, xb=[0,10], yb=[0,10],rand=True,dump_freq=10)
agent = DQN(swimmer, epochs=3200, batch_size=128, gamma=0.98, epsilon_min=0.1, epsilon_decay=0.999)
scores = agent.train(0)
agent.model.save('saved_model/saved_model-%s'%state_size)
plt.plot(scores)

plt.show()

