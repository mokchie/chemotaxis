import pdb
import random
import numpy as np
import tensorflow as tf
import os,sys
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)
del path
from swimmer_v2 import *
os.environ['KMP_DUPLICATE_LIB_OK']='True'
state_size = 4
epsilon = 0.2
N = 40
loaded_model = tf.keras.models.load_model('../chemotaxis3D_nstates-gamma-o2-positive/saved_model/saved_model_sample-DDQN-n%s'%state_size)

#conc_field = Conc_field_radial(c0=200,k=1)
conc_field = Conc_field(c0=200,k=1)
for tau0 in np.linspace(-6.7,6.7,10):
    sname = 'test-DDQN-n%s-tau%.2f'%(state_size,tau0)
    clear(sname)
    random.seed(83843)
    test_swimmer2 = Swimmer(dim=3,
                            v0=2,
                            vw=0.2,
                            k0=6.5, kw=2.0, kn=2,
                            tau0=tau0, tauw=2.0, taun=2,
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
                            xb=[40,50],yb=[40,50],zb=[40,50],
                            rand=True,
                            dump_freq=1,
                            saving_interval_dt=10,
                            actionAll=False)

    agent2 = DQN(test_swimmer2, epochs=100, batch_size=128)
    scores2 = []
    for i in range(N):
        print('test',i)
        state2 = test_swimmer2.reset()
        done2 = False
        R2 = 0
        j=0
        while not done2:
            if random.random() < epsilon:
                ai2 = test_swimmer2.action_space_sample()
            else:
                ai2 = agent2.greedy_action(state2)
            next_state2, reward2, done2, _ = test_swimmer2.step(state2,test_swimmer2.actions[ai2])
                #print(reward)
            R2+=reward2
            state2 = deepcopy(next_state2)
            j+=1
            print('R2:',R2)
            j+=1
            if done2:
                scores2.append(state2[0])
                #print(test_swimmer.rx0-test_swimmer.rx)
                #pdb.set_trace()
