
import tensorflow as tf
from copy import deepcopy
from swimmer import *
os.environ['KMP_DUPLICATE_LIB_OK']='True'
conc_field = Conc_field(c0=20,k=1)
test_swimmer1 = Swimmer(v0=1, k0=4.0, kw=2.0, kn=2, t0=0, rx0=10, ry0=10, tx0=-1, ty0=0, nx0=0, ny0=-1, Taction=1/2, dt=0.02, field=conc_field.get_conc,targetx=0, targety=1000, lifespan=200, state_size=8, sname='test1-sw', xb=[-20,20], yb=[-20,20],rand=True)
test_swimmer2 = Swimmer(v0=1, k0=4.0, kw=2.0, kn=2, t0=0, rx0=10, ry0=10, tx0=-1, ty0=0, nx0=0, ny0=-1, Taction=1/4, dt=0.02, field=conc_field.get_conc,targetx=0, targety=1000, lifespan=200, state_size=8, sname='test2-sw', xb=[-20,20], yb=[-20,20],rand=True)
test_swimmer3 = Swimmer(v0=1, k0=4.0, kw=2.0, kn=2, t0=0, rx0=10, ry0=10, tx0=-1, ty0=0, nx0=0, ny0=-1, Taction=1/8, dt=0.02, field=conc_field.get_conc,targetx=0, targety=1000, lifespan=200, state_size=8, sname='test3-sw', xb=[-20,20], yb=[-20,20],rand=True)
agent1 = DQN(test_swimmer1, epochs=100, batch_size=128)
agent2 = DQN(test_swimmer2, epochs=100, batch_size=128)
agent3 = DQN(test_swimmer3, epochs=100, batch_size=128)
scores1 = []
scores2 = []
scores3 = []
clear('test1-sw')
clear('test2-sw')
clear('test3-sw')
epsilon = 0.0
for i in range(1):
    print('test_sw',i)
    state1 = test_swimmer1.reset()
    done1 = False
    R1 = 0
    state2 = test_swimmer2.reset_copy(test_swimmer1)
    done2 = False
    R2 = 0
    state3 = test_swimmer3.reset_copy(test_swimmer1)
    R3 = 0
    done3 = False
    j = 0
    while (not done1):
        if j%1==0:
            ai1 = agent1.swing_action(state1)
        next_state1, reward1, done1, _ = test_swimmer1.step(state1,test_swimmer1.actions[ai1])
        state1 = deepcopy(next_state1)
        j+=1
    j = 0
    while (not done2):
        if j%2==0:
            ai2 = agent2.swing_action(state2)
        next_state2, reward2, done2, _ = test_swimmer2.step(state2,test_swimmer2.actions[ai2])
        state2 = deepcopy(next_state2)
        j+=1
    j = 0
    while (not done3):
        if j%4==0:
            ai3 = agent3.swing_action(state3)
        next_state3, reward3, done3, _ = test_swimmer3.step(state3, test_swimmer3.actions[ai3])
        state3 = deepcopy(next_state3)
        j+=1


