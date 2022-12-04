import numpy as np
import tensorflow as tf

from swimmer import *
os.environ['KMP_DUPLICATE_LIB_OK']='True'
state_size = 4
sname = 'test'
loaded_model = tf.keras.models.load_model('saved_model/saved_model1')
clear(sname)
conc_field = Conc_field(c0=20,k=1)
test_swimmer1 = Swimmer(v0=1, k0=4.0, kw=2.0, kn=2, t0=0, rx0=2, ry0=10, tx0=1, ty0=0, nx0=0, ny0=-1, Taction=1/state_size, dt=0.02, field=conc_field.get_conc,targetx=0, targety=1000, lifespan=160, state_size=state_size, sname=sname, xb=[0,10], yb=[0,10],rand=True, dump_freq=1, tau0=1.0, tauw=0.2, taun=2, dim=3)

agent1 = DQN(test_swimmer1, epochs=100, batch_size=128)

scores1 = []

for i in range(40):
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
