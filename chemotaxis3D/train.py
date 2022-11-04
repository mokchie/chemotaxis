from swimmer import *
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK']='True'
clear('sample')
random.seed(39895)
conc_field = Conc_field(c0=20,k=1)
swimmer = Swimmer(v0=1, k0=4.0, kw=2.0, kn=2, t0=0, rx0=2, ry0=10, tx0=1, ty0=0, nx0=0, ny0=-1, Taction=1/8, dt=0.02, field=conc_field.get_conc,targetx=0, targety=1000, lifespan=80, state_size=8, sname='sample', xb=[0,10], yb=[0,10],rand=True,dump_freq=1)

agent = DQN(swimmer, epochs=20, batch_size=128, gamma=0.98, epsilon_min=0.1, epsilon_decay=0.98)

scores = agent.train()
agent.model.save('saved_model/saved_model1')

plt.plot(scores)

plt.show()