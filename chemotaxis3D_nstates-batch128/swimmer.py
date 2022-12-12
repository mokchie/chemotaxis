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
    def __init__(self,c0=200,k=1):
        self.c0 = c0
        self.k = k
    def get_conc(self,x,y,z=0):
        # return c0-np.sqrt(x**2+y**2)/2.5
        return self.c0+self.k*y
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
    def __init__(self, dim=3, v0=1.0, k0=1.0, kw=1.0, kn=2, tau0=0, tauw=0, taun=1,t0=0.0, rx0=0.0, ry0=0.0, rz0=0, tx0=1.0, ty0=0.0, tz0=0, nx0=0.0, ny0=1.0, nz0=0, dt=0.002, Taction=1/4, field=Conc_field(), targetx=0.0, targety=10000, targetz=0, lifespan=10, sname='sample', xb=(0,0), yb=(0,0), zb=(0,0), state_size=4, rand=False, dump_freq=1):
        self.sname = sname
        self.epch = 0
        self.v0 = v0
        self.k0 = k0
        self.kw = kw
        self.kn = kn
        self.tau0 = tau0
        self.tauw = tauw
        self.taun = taun
        self.dim = dim
        if self.dim == 2:
            self.lstate = 2
        else:
            self.lstate = 3
        self.dt = dt
        self.Taction = Taction
        self.t = t0
        self.t0 = t0
        self.xb = xb
        self.yb = yb
        self.zb = zb
        self.rx0 = rx0
        self.ry0 = ry0
        self.rz0 = rz0
        self.tx0 = tx0
        self.ty0 = ty0
        self.tz0 = tz0
        self.nx0 = nx0
        self.ny0 = ny0
        self.nz0 = nz0
        self.bx0 = ty0*nz0-tz0*ny0
        self.by0 = tz0*nx0-tx0*nz0
        self.bz0 = tx0*ny0-ty0*nx0
        self.rx = rx0
        self.ry = ry0
        self.rz = rz0
        self.tx = tx0
        self.ty = ty0
        self.tz = tz0
        self.nx = nx0
        self.ny = ny0
        self.nz = nz0
        self.bx = self.bx0
        self.by = self.by0
        self.bz = self.bz0
        self.conc_field = field
        self.get_conc = field.get_conc
        self.targetx = targetx
        self.targety = targety
        self.targetz = targetz
        self.done = False
        self.lifespan = lifespan
        self.state_size = state_size  # the most recent concentration, and the values of 'a'
        if self.dim==2:
            self.actions = list(np.linspace(k0-kw/2,k0+kw/2,self.kn))
            self.kappa = self.actions[int(self.kn/2)]
            self.tau = self.tau0
        else:
            kappa_list = list(np.linspace(k0-kw/2,k0+kw/2,self.kn))
            tau_list = list(np.linspace(tau0-tauw/2,tau0+tauw/2,self.taun))
            self.actions = []
            for ka in kappa_list:
                for tu in tau_list:
                    self.actions.append((ka,tu))
            self.kappa,self.tau = self.actions[int(self.kn*self.taun/2)]
        
        self.action_size = len(self.actions)
        self.rand = rand
        self.dump_freq = dump_freq
        self.memc = deque(maxlen=self.state_size)

    def reset(self,theta=None):
        self.memc = deque(maxlen=self.state_size)
        self.epch += 1
        if self.dim==2:
            self.kappa = self.actions[int(self.kn/2)]
            self.tau = self.tau0
        else:
            self.kappa,self.tau = self.actions[int(self.kn*self.taun/2)]
        if self.epch%self.dump_freq==0:
            self.fp = open('data/%s-epoch-%d.data' % (self.sname, self.epch), 'w')
        self.t = self.t0
        if not self.rand:
            self.rx = self.rx0
            self.ry = self.ry0
            self.rz = self.rz0
            self.tx = self.tx0
            self.ty = self.ty0
            self.tz = self.tz0
            self.nx = self.nx0
            self.ny = self.ny0
            self.nz = self.nz0
            self.bx = self.bx0
            self.by = self.by0
            self.bz = self.bz0
        else:
            if theta is None:
                theta1 = random.random() * 2 * np.pi
                theta2 = random.random() * np.pi
                theta3 = random.random() * 2 * np.pi
                self.rx = random.uniform(self.xb[0], self.xb[1])
                self.ry = random.uniform(self.yb[0], self.yb[1])
                self.rz = random.uniform(self.zb[0], self.zb[1])
                self.rx0 = self.rx
                self.ry0 = self.ry
                self.rz0 = self.rz
            else:
                if self.dim != 2:
                    theta1,theta2,theta3 = theta
                else:
                    theta1 = theta
                self.rx = self.rx0
                self.ry = self.ry0
                self.rz = self.rz0
            if self.dim == 2:
                self.tx, self.ty = np.cos(theta1),np.sin(theta1)
                self.nx = -self.ty
                self.ny = self.tx
                self.tz = 0
                self.nz = 0
            else:
                self.tx, self.ty, self.tz = np.cos(theta2), np.sin(theta1)*np.sin(theta2), -np.cos(theta1)*np.sin(theta2)
                self.nx, self.ny, self.nz = np.sin(theta2)*np.sin(theta3), np.cos(theta1)*np.cos(theta3) - np.cos(theta2)*np.sin(theta1)*np.sin(theta3), np.cos(theta3)*np.sin(theta1) + np.cos(theta1)*np.cos(theta2)*np.sin(theta3)

            self.bx = self.ty*self.nz-self.tz*self.ny
            self.by = self.tz*self.nx-self.tx*self.nz
            self.bz = self.tx*self.ny-self.ty*self.nx
            self.tx0 = self.tx
            self.ty0 = self.ty
            self.tz0 = self.tz
            self.nx0 = self.nx
            self.ny0 = self.ny
            self.nz0 = self.nz
            self.bx0 = self.bx
            self.by0 = self.by
            self.bz0 = self.bz
            


        self.done = False
        # self.ax.clear()
        if self.epch%self.dump_freq==0:
            if self.dim==2:
                self.fp.write('%f %f %f %f %f %f %f %f\n' % (self.t, self.rx, self.ry, self.tx, self.ty, self.nx, self.ny, self.kappa))
            else:
                self.fp.write('%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n' % (self.t, self.rx, self.ry, self.rz, self.tx, self.ty, self.tz, self.nx, self.ny, self.nz, self.bx, self.by, self.bz, self.kappa, self.tau))
        c = self.get_conc(self.rx, self.ry, self.rz)
        self.memc.append(c)
        if self.dim == 2:
            states = np.array([c,self.kappa] * self.state_size)
        else:
            states = np.array([c, self.kappa, self.tau] * self.state_size)
        return states
    def reset_copy(self,swimmerc):
        self.memc = deque(maxlen=self.state_size)
        self.epch += 1
        if self.dim==2:
            self.kappa = self.actions[int(self.kn/2)]
            self.tau = self.tau0
        else:
            self.kappa,self.tau = self.actions[int(self.kn*self.taun/2)]           
        if self.epch%self.dump_freq==0:
            self.fp = open('data/%s-epoch-%d.data' % (self.sname, self.epch), 'w')
        self.t = self.t0
        self.rx = swimmerc.rx
        self.ry = swimmerc.ry
        self.rz = swimmerc.rz
        self.rx0 = swimmerc.rx0
        self.ry0 = swimmerc.ry0
        self.rz0 = swimmerc.rz0
        self.tx, self.ty, self.tz = swimmerc.tx, swimmerc.ty, swimmerc.tz
        self.tx0, self.ty0, self.tz0 = swimmerc.tx0, swimmerc.ty0, swimmerc.tz0
        self.nx, self.ny, self.nz = swimmerc.nx, swimmerc.ny, swimmerc.nz
        self.nx0, self.ny0, self.nz0 = swimmerc.nx0, swimmerc.ny0, swimmerc.nz0
        self.bx, self.by, self.bz = swimmerc.bx, swimmerc.by, swimmerc.bz
        self.bx0, self.by0, self.bz0 = swimmerc.bx0, swimmerc.by0, swimmerc.bz0
        self.done = False
        # self.ax.clear()
        if self.epch%self.dump_freq==0:
            if self.dim==2:
                self.fp.write('%f %f %f %f %f %f %f %f\n' % (self.t, self.rx, self.ry, self.tx, self.ty, self.nx, self.ny, self.kappa))
            else:
                self.fp.write('%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n' % (self.t, self.rx, self.ry, self.rz, self.tx, self.ty, self.tz, self.nx, self.ny, self.nz, self.bx, self.by, self.bz, self.kappa, self.tau))

        c = self.get_conc(self.rx, self.ry)
        self.memc.append(c)
        if self.dim == 2:
            states = np.array([c,self.kappa] * self.state_size)
        else:
            states = np.array([c, self.kappa, self.tau] * self.state_size)
        return states
    def spermtrj(self,mu,rho,filename=None):
        '''this sperm trj only works in 3D.'''
        T = [0,]
        X = [self.rx0,]
        Y = [self.ry0,]
        Z = [self.rz0,]
        rx = self.rx0
        ry = self.ry0
        rz = self.rz0
        tx = self.tx0
        ty = self.ty0
        tz = self.tz0
        nx = self.nx0
        ny = self.ny0
        nz = self.nz0
        bx = self.bx0
        by = self.by0
        bz = self.bz0

        a = 1
        if self.dim == 2:
            p = 1/self.get_conc(rx, ry)
        else:
            p = 1/self.get_conc(rx, ry, rz)
        t = 0
        while t<=self.lifespan:
            t+=self.dt
            a0 = a
            p0 = p
            s = self.get_conc(rx,ry,rz)
            if self.dim == 2:
                kappa = self.k0-rho*self.k0*(a0-1)
                F = np.matrix([[tx,nx,rx],[ty,ny,ry],[0,0,1]])
                A = np.matrix([[0,-kappa*self.v0,self.v0],[kappa*self.v0,0,0],[0,0,0]])
                F = F * matrixexp(A * dt)
                rx, ry, tx, ty, nx, ny = F[0,2],F[1,2],F[0,0],F[1,0],F[0,1],F[1,1]
            else:
                kappa = self.k0-rho*self.k0*(a0-1)                
                tau = self.tau0+rho*self.tau0*(a0-1)
                F = np.matrix([[tx, nx, bx, rx], [ty, ny, by, ry], [tz, nz, bz,  rz], [0, 0, 0, 1]])
                A = np.matrix([[0, -kappa * self.v0, 0, self.v0], [kappa * self.v0, 0, -tau * self.v0, 0], [0, tau * self.v0, 0, 0], [0, 0, 0, 0]])
                F = F * matrixexp(A * dt)
                rx, ry, rz = F[0, 3], F[1, 3], F[2, 3]
                bx, by, bz = F[0, 2], F[1, 2], F[2, 2]
                tx, ty, tz = F[0, 0], F[1, 0], F[2, 0]
                nx, ny, nz = F[0, 1], F[1, 1], F[2, 1]

            a = a0 + self.dt*(p0*s-a0)/mu
            p = p0 + self.dt*p0*(1-a0)/mu
            T.append(t)
            X.append(rx)
            Y.append(ry)
            Z.append(rz)
        
        with open('data/'+filename,'w') as ftrj:
            for t,rx,ry,rz in zip(T,X,Y,Z):
                ftrj.write('%s %s %s\n'%(t,rx,ry,rz))
        if self.dim==2:
            return np.array(T),np.array(X),np.array(Y)
        else:
            return np.array(T),np.array(X),np.array(Y),np.array(Z)
            
    def step(self, states, target):
        if self.dim == 2:
            omega = self.v0*self.k0
        else:
            omega = self.v0*np.sqrt(self.k0**2+self.tau0**2)
        ntimes = int(np.round(self.Taction * 2 * np.pi / omega / self.dt))
        dt = self.Taction * 2 * np.pi / omega / ntimes
        previous_states = deepcopy(states[0:-self.lstate])
        for i in range(ntimes):
            if self.dim == 2:
                self.kappa = target
                kappa = self.kappa
                F = np.matrix([[self.tx,self.nx,self.rx],[self.ty,self.ny,self.ry],[0,0,1]])
                A = np.matrix([[0,-kappa*self.v0,self.v0],[kappa*self.v0,0,0],[0,0,0]])
                F = F * matrixexp(A * dt)
                self.rx, self.ry, self.tx, self.ty, self.nx, self.ny = F[0,2],F[1,2],F[0,0],F[1,0],F[0,1],F[1,1]
            else:
                self.kappa,self.tau = target
                kappa = self.kappa
                tau = self.tau
                F = np.matrix([[self.tx, self.nx, self.bx, self.rx], [self.ty, self.ny, self.by, self.ry], [self.tz, self.nz, self.bz,  self.rz], [0, 0, 0, 1]])
                A = np.matrix([[0, -kappa * self.v0, 0, self.v0], [kappa * self.v0, 0, -tau * self.v0, 0], [0, tau * self.v0, 0, 0], [0, 0, 0, 0]])
                F = F * matrixexp(A * dt)
                self.rx, self.ry, self.rz = F[0, 3], F[1, 3], F[2, 3]
                self.bx, self.by, self.bz = F[0, 2], F[1, 2], F[2, 2]
                self.tx, self.ty, self.tz = F[0, 0], F[1, 0], F[2, 0]
                self.nx, self.ny, self.nz = F[0, 1], F[1, 1], F[2, 1]

            self.t += dt
            if self.epch%self.dump_freq==0:
                if self.dim == 2:
                    self.fp.write('%f %f %f %f %f %f %f %f\n' % (
                    self.t, self.rx, self.ry, self.tx, self.ty, self.nx, self.ny, self.kappa))
                else:
                    self.fp.write('%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n' % (
                    self.t, self.rx, self.ry, self.rz, self.tx, self.ty, self.tz, self.nx, self.ny, self.nz, self.bx,
                    self.by, self.bz, self.kappa, self.tau))
            if self.dim==2:
                disp1 = np.sqrt((self.rx - self.targetx) ** 2 + (self.ry - self.targety) ** 2)
            else:
                disp1 = np.sqrt((self.rx - self.targetx) ** 2 + (self.ry - self.targety) ** 2 + (self.rz - self.targetz) ** 2)
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
        # print('(x,y)',self.rx,self.ry)

        c = self.get_conc(self.rx, self.ry)
        ca0 = np.average(self.memc)
        self.memc.append(c)
        ca1 = np.average(self.memc)
        #reward = (c_center1 - c_center0)*10
        if self.dim==2:
            reward = (ca1-ca0)/np.abs(1/(self.k0-self.kw/2) - 1/(self.k0+self.kw/2))/self.conc_field.k
        else:
            rmin = (self.k0+self.kw/2)/((self.k0+self.kw/2)**2+(self.tau0+self.tauw/2)**2)
            rmax = (self.k0-self.kw/2)/((self.k0-self.kw/2)**2+(self.tau0-self.tauw/2)**2)
            reward = (ca1 - ca0) / np.abs(rmax-rmin) / self.conc_field.k
        #pdb.set_trace()
        if self.dim == 2:
            return [np.concatenate((np.array([c,self.kappa]),previous_states)), reward, self.done, {}]
        else:
            return [np.concatenate((np.array([c, self.kappa, self.tau]), previous_states)), reward, self.done, {}]


    def action_space_sample(self):
        return random.randrange(self.action_size)

class DQN():
    def __init__(self, swimmer=Swimmer(), epochs=100, batch_size=128, gamma=0.98, epsilon_min=0.1, epsilon_decay=0.98):
        self.memory = deque(maxlen=10000)
        self.env = swimmer
        self.input_size = self.env.state_size*self.env.lstate
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
        states_out = []
        if self.env.dim == 2:
            sa1 = np.average(states[0::self.env.lstate])
            sa2 = self.env.k0
            for i,s in enumerate(states):
                if i%self.env.lstate==0:
                    states_out.append((s-sa1)*self.env.k0/self.env.conc_field.k)
                else:
                    states_out.append((s-sa2)/(self.env.kw/2))
        else:
            sa1 = np.average(states[0::self.env.lstate])
            sa2 = self.env.k0
            sa3 = self.env.tau0
            for i,s in enumerate(states):
                if i%self.env.lstate==0:
                    states_out.append((s-sa1)*self.env.k0/self.env.conc_field.k)
                elif (i-1)%self.env.lstate==0:
                    states_out.append((s-sa2)/(self.env.kw/2))
                else:
                    states_out.append((s-sa3)/(self.env.tauw/2))
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

    def human_choose_action(self,states):
        if self.env.dim != 2:
            print('greedy strategy only works for 2d')
            raise TypeError
        ss = self.preprocess(states)
        nmax = np.argmax(ss[0::self.env.lstate])
        nmin = np.argmin(ss[0::self.env.lstate])
        #pdb.set_trace()
        if nmax == 0:
            return -1
        elif nmin == 0:
            return 0
        else:
            return self.env.actions.index(states[1])
    def swing_action(self,states):
        if self.env.dim == 2:
            return self.env.action_size - self.env.actions.index(states[1]) - 1
        else:
            return self.env.action_size - self.env.actions.index((states[1],states[2])) - 1

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

    def train(self):
        self.frw = open('data/%s-rewards.data' % (self.env.sname,), 'w')
        scores = deque(maxlen=8)
        avg_scores = []
        for e in range(self.epochs):
            print('epoch ', e)
            states = self.env.reset()
            done = False
            rets = []
            tot_reward = 0
            i = 1
            while not done:
                #pdb.set_trace()
                if i < self.env.state_size:
                    ai = self.choose_rand_action()
                    next_states, reward, done, _ = self.env.step(states, self.env.actions[ai])
                else:
                    ai = self.choose_action(states, self.epsilon)
                    next_states, reward, done, _ = self.env.step(states, self.env.actions[ai])
                    self.remember(deepcopy(states), ai, reward, next_states, done)
                    #pdb.set_trace()

                    # self.epsilon = max(self.epsilon_min, self.epsilon_decay*self.epsilon) # decrease epsilon
                states = next_states
                i += 1
                rets.append(reward)
                tot_reward+=reward
            ret_sum = 0
            for ret in reversed(rets):
                ret_sum = ret+self.gamma*ret_sum
            if self.env.dim == 2:
                scores.append(self.env.get_conc(self.env.rx,self.env.ry) - self.env.get_conc(self.env.rx0,self.env.ry0))
            else:
                scores.append(self.env.get_conc(self.env.rx, self.env.ry, self.env.rz) - self.env.get_conc(self.env.rx0, self.env.ry0, self.env.rz0))
            self.frw.write('%s %s %s\n' % (e, ret_sum, tot_reward))
            mean_score = np.mean(scores)
            print(f'tot_reward: {tot_reward} return: {ret_sum} mean_score: {mean_score}' )
            avg_scores.append(mean_score)
            self.replay(self.batch_size)
        self.frw.close()
        print('Run {} episodes'.format(e))
        return avg_scores
