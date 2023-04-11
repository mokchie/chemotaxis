from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.optimizers import Adam
import os,sys,pdb,random
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if not path in sys.path:
    sys.path.insert(1, path)
del path
from swimmer_v2 import *
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
matplotlib.rcParams.update({'font.size':14, 'font.family':'sans-serif'})

fig,ax = plt.subplots(1,1)
os.environ['KMP_DUPLICATE_LIB_OK']='True'
conc_field = Conc_field(c0=20,k=1)
colors = ['C0','C1']
def vector2radian(vec):
    theta = np.arccos(np.dot(vec,np.array([0,1,0])))
    phi = np.arctan2(vec[2],vec[0])
    return (theta,phi)
def radian2vector(radian):
    theta,phi = radian
    y = np.cos(theta)
    x = np.sin(theta)*np.cos(phi)
    z = np.sin(theta)*np.sin(phi)
    return (x,y,z)
def scale_theta(theta):
    return (theta/np.pi-0.5)*2
ft = open('./data/result.data','wb')
for cn,state_size in enumerate([2,4]):
    sname = 'sample-NN-n%s'%state_size
    random.seed(39895)
    kappa0 = 6.5
    tau0 = 6.7
    swimmer = Swimmer(dim=3,
                      v0=2,
                      vw=0.2,
                      k0=kappa0, kw=2, kn=2,
                      tau0=tau0, tauw=2, taun=2,
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
                      dump_freq=1000,
                      saving_interval_dt=10,
                      actionAll=False)
    def create_dataset(n):
        X = []
        Y = []
        for i in range(n):
            print(i)
            states = swimmer.reset()
            for j in range(state_size):
                ai = swimmer.action_space_sample()
                action = swimmer.actions[ai]
                states,reward,done,_ = swimmer.step(states,action)
            x = swimmer.preprocess(states)
            print(x)
            dirc_vec = swimmer.get_h()
            theta,phi = vector2radian(dirc_vec)
            theta2 = np.arccos(np.dot(np.array([swimmer.nx,swimmer.ny,swimmer.nz]),np.cross(dirc_vec,np.array([0,1,0]))))
            y = np.array([scale_theta(theta),scale_theta(theta2)])
            print(y)
            X.append(x)
            Y.append(y)
        return np.array(X),np.array(Y)
    
    data_file = 'data/train_data-nt%s.data'%state_size
    if os.path.isfile(data_file) and len(sys.argv)==1:
        with open(data_file,'rb') as fp:
            x_ds = np.load(fp)
            y_ds = np.load(fp)
            split = int(len(x_ds)*0.8)
            x_train = x_ds[0:split]
            y_train = y_ds[0:split]
            x_test = x_ds[split:]
            y_test = y_ds[split:]
    else:
        clear(sname)        
        x_ds, y_ds = create_dataset(50000)
        split = int(len(x_ds)*0.8)
        x_train = x_ds[0:split]
        y_train = y_ds[0:split]
        x_test = x_ds[split:]
        y_test = y_ds[split:]    
        with open(data_file,'wb') as fp:
            np.save(fp,x_ds)
            np.save(fp,y_ds)
    model = Sequential()
    model.add(Dense(32, input_dim=state_size*2, activation='tanh'))
    for j in range(3):
        model.add(Dense(32, activation='tanh'))
    model.add(Dense(2, activation='tanh'))
    model.compile(loss=tf.keras.losses.mse, # mae is short for mean absolute error
                  optimizer=Adam(learning_rate=0.01, decay=0.1),
                  metrics=["mse"])
    Ep = []
    Ave = []
    Std = []
    for i in range(100):
        model.fit(x_train, y_train, batch_size=32, epochs=1)
        if i%5==0:
            y_pred = model.predict(x_test)
            avei = np.average(np.abs(y_pred-y_test),axis=0)
            print(avei)
            stdi = np.std(np.abs(y_pred-y_test),axis=0) 
            print(stdi)
            Ep.append(i)
            Ave.append(avei)
            Std.append(stdi)
    Ep = np.array(Ep)
    Ave = np.array(Ave)/2*180
    Std = np.array(Std)/2*180
    np.save(ft,Ep)
    np.save(ft,Ave)
    np.save(ft,Std)
    ax.plot(Ep,Ave[:,0],color=colors[cn],label=r'$N_T=%s$'%state_size)
    ax.fill_between(Ep,Ave[:,0]-Std[:,0],Ave[:,0]+Std[:,0], color=colors[cn],alpha=0.2)
ft.close()
ax.set_xlabel('epoch')
ax.set_ylabel('Error (degree)')
ax.legend()
plt.show()


