import pdb

import numpy as np
import tensorflow as tf
import os,sys
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)
del path
from swimmer_v2 import *
os.environ['KMP_DUPLICATE_LIB_OK']='True'
state_size = 2
sname = 'testone-n%s-swinging'%state_size
#conc_field = Conc_field_radial(c0=200,k=1)
conc_field = Conc_field(c0=200,k=1)
test_swimmer1 = Swimmer(dim=3,
                      v0=2,
                      vw=0.2,
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
                      dump_freq=1,
                      saving_interval_dt=10,
                      actionAll=False)
agent1 = DQN(test_swimmer1, epochs=100, batch_size=128)

scores1 = []

for i in range(1):
    print('testone',i)
    state1 = test_swimmer1.reset()
    done1 = False
    R1 = 0
    j=0
    while not done1:
        #if j<state_size:
        #    ai1 = agent1.greedy_action(state1) #use greedy strategy
        #else:

        #ai1 =test_swimmer1.action_space_sample()
        ai1 = agent1.swing_action(state1)
        next_state1, reward1, done1, _ = test_swimmer1.step(state1,test_swimmer1.actions[ai1])
        #print(reward)
        R1+=reward1
        state1 = deepcopy(next_state1)
        j+=1
        print('R1:',R1)

        if done1:
            scores1.append(state1[0])
            #print(test_swimmer.rx0-test_swimmer.rx)
            #pdb.set_trace()
