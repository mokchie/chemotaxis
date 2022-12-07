import numpy as np
import tensorflow as tf

from swimmer import *
os.environ['KMP_DUPLICATE_LIB_OK']='True'
state_size = 8
sname = 'test'
loaded_model = tf.keras.models.load_model('saved_model/saved_model1')
clear(sname)
#conc_field = Conc_field_radial(c0=200,k=1)
conc_field = Conc_field(c0=200,k=1)

test_swimmer1 = Swimmer(v0=1,
                  k0=6.5, kw=6.5, kn=5,
                  tau0=6.7, tauw=6.7, taun=5,
                  t0=0,
                  rx0=2, ry0=10, rz0=4,
                  tx0=1, ty0=0, tz0=0,
                  nx0=0, ny0=-1, nz0=0,
                  Taction=2/state_size,
                  dt=0.005,
                  field=conc_field,
                  targetx=0, targety=2000, targetz=0,
                  lifespan=160,
                  state_size=state_size,
                  Regg=0.5,
                  sname=sname,
                  xb=[10,20], yb=[10,20],zb=[10,20],
                  rand=True,
                  dump_freq=1,
                  dim=3)

agent1 = DQN(test_swimmer1, epochs=100, batch_size=128)

scores1 = []

for i in range(10):
    print('test',i)
    state1 = test_swimmer1.reset()
    done1 = False
    R1 = 0
    j=0
    while not done1:
        #if j<state_size:
        #    ai1 = agent1.greedy_action(state1) #use greedy strategy
        #else:

        if j<state_size:
            ai1 =test_swimmer1.action_space_sample()
        else:
            ai1 = np.argmax(loaded_model.predict(np.array([agent1.preprocess(state1), ])))
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
