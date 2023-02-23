import numpy as np
import tensorflow as tf
import os,sys
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)
del path
from swimmer_v4 import *

os.environ['KMP_DUPLICATE_LIB_OK']='True'
state_size = 4
epsilon = 0.0
sname = 'test-n%s-epsilon%s'%(state_size,epsilon)
loaded_model = tf.keras.models.load_model('saved_model/saved_model_sample-n%s'%state_size)
clear(sname+'-greedy')
clear(sname+'-DRL')
conc_field = Conc_field(c0=20,k=1)

test_swimmer2 = Swimmer(dim=2,
                  v0=2,
                  k0=6.5, kw=2.0, kn=2,
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
                  sname=sname+'-greedy',
                  xb=[40,50],yb=[40,50],
                  rand=True,
                  dump_freq=1,
                  Regg=1.0,
                  actionAll=False)

test_swimmer3 = Swimmer(dim=2,
                  v0=2,
                  k0=6.5, kw=2.0, kn=2,
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
                  sname=sname+'-DRL',
                  xb=[40,50],yb=[40,50],
                  rand=True,
                  dump_freq=1,
                  Regg=1.0,
                  actionAll=False)

agent2 = DQN(test_swimmer2, epochs=100, batch_size=128)
agent3 = DQN(test_swimmer3, epochs=100, batch_size=128)
scores2 = []
scores3 = []

for i in range(40):
    print('test',i)
    state2 = test_swimmer2.reset()
    done2 = False
    R2 = 0
    state3 = test_swimmer3.reset_copy(test_swimmer2)
    done3 = False
    R3 = 0
    j=0
    while (not done2) and (not done3):
        if random.random() < epsilon:
            ai2 = test_swimmer2.action_space_sample()
        else:
            ai2 = agent2.greedy_action(state2)
        next_state2, reward2, done2, _ = test_swimmer2.step(state2, test_swimmer2.action_labels[ai2])
        R2+=reward2
        state2 = deepcopy(next_state2)

        if j<state_size:
            ai3 = agent3.greedy_action(state3)
        else:
            if random.random() < epsilon:
                ai3 = test_swimmer3.action_space_sample()
            else:
                ai3 = np.argmax(loaded_model.predict(np.array([test_swimmer3.preprocess(state3), ])))
        next_state3, reward3, done3, _ = test_swimmer3.step(state3,test_swimmer3.action_labels[ai3])
        R3+=reward3
        state3 = deepcopy(next_state3)

        j+=1
        print('R2:',R2,'R3:',R3)

        if done2 or done3:
            scores2.append(state2[0])
            scores3.append(state3[0])
