import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.optimizers import Adam
import re,os,sys
import random
import pdb
from copy import deepcopy
eps = 0.2
class Conc_field:
    def __init__(self,c0,k):
        self.c0 = c0
        self.k = k
    def get_conc(self,x,y):
        # return c0-np.sqrt(x**2+y**2)/2.5
        return self.c0+self.k*y
class External_flow_field:
    def __init__(self,u0,k):
        self.u0 = u0
        self.k = k
    def external_vel(self,x,y):
        #k=2*np.pi/20
        #u0 = 0.2
        return [self.u0*np.cos(self.k*x)*np.sin(self.k*y), -self.u0*np.sin(self.k*x)*np.cos(self.k*y)]
    def external_rotational_vel(self,x,y):
        #k=2*np.pi/20
        #u0 = 0.2
        return -self.u0*self.k*np.cos(self.k*x)*np.cos(self.k*y)    
def matrixexp(A):
    n,m = A.shape
    E = np.matrix(np.eye(n,m))
    A2 = A*A
    A3 = A2*A
    A4 = A3*A
    A5 = A4*A
    num = E+A/2+A2/9+A3/72+A4/1008+A5/30240
    den = E-A/2+A2/9-A3/72+A4/1008-A5/30240
    return num*den.I

def clear(sname):
    pattern = re.compile("^%s-epoch-([0-9]+).data$"%sname)
    for root,dirs,files in os.walk("data"):
        for name in files:
            if pattern.match(name):
                print(name + ' removed')
                os.remove('data/'+name)

class Swimmer:
    def __init__(self, v0, k0, kw, kn, t0, rx0, ry0, tx0, ty0, nx0, ny0, dt, Taction, field, targetx, targety, lifespan, sname, xb, yb, state_size, vw=None, rand=False, dump_freq=1):
        self.sname = sname
        self.epch = 0
        self.v0 = v0
        self.vw = vw
        self.k0 = k0
        self.kw = kw
        self.kn = kn
        self.dt = dt
        self.Taction = Taction
        self.t = t0
        self.t0 = t0
        self.xb = xb
        self.yb = yb
        self.rx0 = rx0
        self.ry0 = ry0
        self.tx0 = tx0
        self.ty0 = ty0
        self.nx0 = nx0
        self.ny0 = ny0
        self.rx = rx0
        self.ry = ry0
        self.tx = tx0
        self.ty = ty0
        self.nx = nx0
        self.ny = ny0
        self.conc_field = field
        self.targetx = targetx
        self.targety = targety
        self.done = False
        self.lifespan = lifespan
        self.state_size = state_size  # the most recent concentration, and the values of 'a'
        self.actions = list(np.linspace(k0-kw/2,k0+kw/2,self.kn))
        if self.vw:
            self.vs = list(np.linspace(self.v0+self.vw/2,self.v0-self.vw/2,self.kn))
        else:
            self.vs = [self.v0,]*self.kn
        self.kappa = self.actions[int(self.kn/2)]
        self.v = self.vs[int(self.kn/2)]
        self.action_size = len(self.actions)
        self.rand = rand
        self.dump_freq = dump_freq
        self.memc = deque(maxlen=self.state_size)

    def reset(self,theta=None):
        self.memc = deque(maxlen=self.state_size)
        self.epch += 1
        self.kappa = self.actions[int(self.kn/2)]
        self.v = self.vs[int(self.kn / 2)]
        if self.epch%self.dump_freq==0:
            self.fp = open('data/%s-epoch-%d.data' % (self.sname, self.epch), 'w')
        self.t = self.t0
        if not self.rand:
            self.rx = self.rx0
            self.ry = self.ry0
            self.tx = self.tx0
            self.ty = self.ty0
            self.nx = self.nx0
            self.ny = self.ny0
        else:
            if theta is None:
                theta = random.random() * 2 * np.pi
                self.rx = random.uniform(self.xb[0], self.xb[1])
                self.ry = random.uniform(self.yb[0], self.yb[1])
                self.rx0 = self.rx
                self.ry0 = self.ry
            else:
                self.rx = self.rx0
                self.ry = self.ry0
            self.tx, self.ty = np.cos(theta),np.sin(theta)
            self.nx = -self.ty
            self.ny = self.tx


        self.done = False
        # self.ax.clear()
        if self.epch%self.dump_freq==0:
            self.fp.write('%f %f %f %f %f %f\n' % (self.t, self.rx, self.ry, self.nx, self.ny, self.kappa))
        c = self.conc_field(self.rx, self.ry)
        self.memc.append(c)
        states = np.array([c,self.kappa] * self.state_size)
        return states
    def reset_copy(self,swimmerc):
        self.memc = deque(maxlen=self.state_size)
        self.epch += 1
        self.kappa = self.actions[int(self.kn/2)]
        self.v = self.vs[int(self.kn / 2)]
        if self.epch%self.dump_freq==0:
            self.fp = open('data/%s-epoch-%d.data' % (self.sname, self.epch), 'w')
        self.t = self.t0
        self.rx = swimmerc.rx
        self.ry = swimmerc.ry
        self.rx0 = self.rx
        self.ry0 = self.ry
        self.tx, self.ty = swimmerc.tx, swimmerc.ty
        self.tx0, self.ty0 = self.tx, self.ty
        self.nx, self.ny = swimmerc.nx, swimmerc.ny
        self.nx0, self.ny0 = self.nx, self.ny
        self.done = False
        # self.ax.clear()
        if self.epch%self.dump_freq==0:
            self.fp.write('%f %f %f %f %f %f\n' % (self.t, self.rx, self.ry, self.nx, self.ny, self.kappa))
        c = self.conc_field(self.rx, self.ry)
        self.memc.append(c)
        states = np.array([c,self.kappa,] * self.state_size)
        return states

    def step(self, states, target_kappa):
        self.v = self.vs[self.actions.index(target_kappa)]
        ntimes = int(np.round(self.Taction * 2 * np.pi / self.k0 / self.v0 / self.dt))
        dt = self.Taction * 2 * np.pi / self.k0 / self.v0 / ntimes
        previous_states = deepcopy(states[0:-2])
        for i in range(ntimes):
            self.kappa = target_kappa
            kappa = self.kappa
            F = np.matrix([[self.tx,self.nx,self.rx],[self.ty,self.ny,self.ry],[0,0,1]])
            A = np.matrix([[0,-kappa*self.v,self.v],[kappa*self.v,0,0],[0,0,0]])
            F = F * matrixexp(A * dt)
            self.rx, self.ry, self.tx, self.ty, self.nx, self.ny = F[0,2],F[1,2],F[0,0],F[1,0],F[0,1],F[1,1]

            self.t += dt
            if self.epch%self.dump_freq==0:
                self.fp.write('%f %f %f %f %f %f\n' % (self.t, self.rx, self.ry, self.nx, self.ny, self.kappa))
            disp1 = np.sqrt((self.rx - self.targetx) ** 2 + (self.ry - self.targety) ** 2)
            if disp1 < eps:
                self.done = True
                if self.epch%self.dump_freq==0:
                    self.fp.close()
                break
            elif self.t > self.lifespan:
                self.done = True
                if self.epch%self.dump_freq==0:
                    self.fp.close()
                break
            else:
                self.done = False
        cx, cy = self.rx + self.nx / target_kappa, self.ry + self.ny / target_kappa
        c = self.conc_field(self.rx, self.ry)
        ca0 = np.average(self.memc)
        self.memc.append(c)
        ca1 = np.average(self.memc)
        #reward = (c_center1 - c_center0)*10
        reward = (ca1-ca0)/np.abs(1/(self.k0-self.kw/2) - 1/(self.k0+self.kw/2))
        #pdb.set_trace()
        return [np.concatenate((np.array([c,self.kappa]),previous_states)), reward, self.done, {}]


    def action_space_sample(self):
        return random.randrange(self.action_size)

class DQN():
    def __init__(self, swimmer, epochs, batch_size=128, gamma=0.98, epsilon_min=0.1, epsilon_decay=0.98):
        self.memory = deque(maxlen=10000)
        self.env = swimmer
        self.input_size = self.env.state_size*2
        self.action_size = self.env.action_size
        self.batch_size = batch_size
        self.gamma = gamma #gamma is the learning rate
        self.epsilon = 1.0
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.epochs = epochs
        # self.hist_size = 1
        alpha = 0.01
        alpha_decay = 0.1

        # Init model
        self.model = Sequential()
        #        self.model.add(keras.layers.LSTM(16,input_shape=(None,self.input_size)))
        self.model.add(Dense(24, input_dim=self.input_size, activation='tanh'))
        #self.model.add(Dropout(0.3))
        self.model.add(Dense(24, activation='tanh'))
        #self.model.add(Dropout(0.3))
        self.model.add(Dense(24, activation='tanh'))
        self.model.add(Dense(self.action_size, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(learning_rate=alpha, decay=alpha_decay))
    def preprocess(self,states):
        sa1 = np.average(states[0::2])
        sa2 = self.env.k0
        states_out = []
        for i,s in enumerate(states):
            if i%2==0:
                states_out.append((s-sa1)*self.env.k0)
            else:
                states_out.append((s-sa2)/(self.env.kw/2))
        return np.array(states_out)
    def remember(self, states, action, reward, next_states, done):
        self.memory.append((states, action, reward, next_states, done))

    def choose_action(self, states, epsilon):
        if np.random.random() <= epsilon:
            return self.env.action_space_sample()
        else:
            return np.argmax(self.model.predict(np.array([self.preprocess(states), ])))

    def choose_rand_action(self):
        return self.env.action_space_sample()

    def greedy_action(self,states):
        ss = self.preprocess(states)
        nmax = np.argmax(ss[0::2])
        nmin = np.argmin(ss[0::2])
        #pdb.set_trace()
        if nmax == 0:
            return -1
        elif nmin == 0:
            return 0
        else:
            return self.env.actions.index(states[1])
    def swing_action(self,states):
        return (self.env.actions.index(states[1])+1)%2

    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        for states, ai, reward, next_states, done in minibatch:
            #pdb.set_trace()
            y_target = self.model.predict(np.array([self.preprocess(states), ]))

            y_target[0][ai] = reward if done else reward + self.gamma * np.max(
                self.model.predict(np.array([self.preprocess(next_states), ]))[0])
            # pdb.set_trace()
            x_batch.append(self.preprocess(states))
            y_batch.append(y_target[0])

        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=2)
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)  # decrease epsilon

    def train(self,super_steps):
        scores = deque(maxlen=8)
        avg_scores = []
        for e in range(self.epochs):
            print('epoch ', e)
            states = self.env.reset()
            done = False
            i = 1
            while not done:
                #pdb.set_trace()
                if i < self.env.state_size:
                    ai = self.choose_rand_action()
                    next_states, reward, done, _ = self.env.step(states, self.env.actions[ai])
                else:
                    if self.env.epch < super_steps:
                        ai = self.human_choose_action(states)
                    else:
                        ai = self.choose_action(states, self.epsilon)
                    if ai is None:
                        next_states, reward, done, _ = self.env.step(states, self.env.kappa)
                    else:
                        next_states, reward, done, _ = self.env.step(states, self.env.actions[ai])
                    self.remember(deepcopy(states), ai, reward, next_states, done)
                    #pdb.set_trace()

                    # self.epsilon = max(self.epsilon_min, self.epsilon_decay*self.epsilon) # decrease epsilon
                states = next_states
                i += 1

            scores.append(self.env.conc_field(self.env.rx,self.env.ry) - self.env.conc_field(self.env.rx0,self.env.ry0))

            mean_score = np.mean(scores)
            # print('epoch ',e)
            print('mean_score ', mean_score)
            avg_scores.append(mean_score)
            self.replay(self.batch_size)

        print('Run {} episodes'.format(e))
        return avg_scores
