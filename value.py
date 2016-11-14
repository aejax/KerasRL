import numpy as np
import gym
import keras.backend as K
from keras.models import Model, Sequential, load_model
from keras.layers import Convolution2D, Dense, Flatten, Input, merge
from keras.preprocessing import image
from sklearn.neighbors import KNeighborsRegressor
import types
from collections import deque

def get_space_dim(Space):
    # get the dimensions of the spaces
    if type(Space) == gym.spaces.Box:
        s_dim = 1
        for n in Space.shape:
            s_dim *= n
    elif type(Space) == gym.spaces.Discrete:
        s_dim = Space.n
    else:
        print 'Wrong type for space: must be either Box or Discrete'
        
    return s_dim

class ValueFunction(object):
    def __init__(self, name=None, lr=1e-3, update_mode='update'):
        if name == None:
            self.name = self.__class__.__name__

        # create a constant function if not a function
        if type(lr) != types.FunctionType:
            self.lr = lambda n: lr

        self.update_mode = update_mode

    def __call__(self, state, *args):
        return self.call(state, *args)

    def call(self, state):
        return state

    def update(self, update, n=None, *args):
        self.lr(n)*update

class KerasQ(ValueFunction):
    def __init__(self, S=None, A=None, model=None, loss='mse', optimizer='adam', bounds=False, batch_size=32, **kwargs):
        self.S = S
        self.A = A

        if model == None:
            assert self.S != None and self.A != None, 'You must either specify a model or as state and action space.'
            self.s_dim = get_space_dim(self.S)
            self.a_dim = get_space_dim(self.A)
            
            self.model = Sequential()
            self.model.add(Dense(100, activation='tanh', input_dim=self.s_dim))
            self.model.add(Dense(self.a_dim))
        else:
            self.model = model

        if bounds:
            out_shape = self.model.outputs[0]._keras_shape
            Umin = Input(shape=(out_shape[-1],), name='Umin')
            Lmax = Input(shape=(out_shape[-1],), name='Lmax')
            output = merge(self.model.outputs+[Umin,Lmax], mode=lambda l: l[0] + 1e-6*l[1] + 1e-6*l[2], output_shape=(out_shape[-1],))
            self.model = Model(input=self.model.inputs+[Umin,Lmax], output=output)

            def bounded_loss(y_true, y_pred):
                penalty = 4
                mse = K.square(y_pred - y_true)
                lb = penalty*K.relu(K.square(Lmax - y_pred))
                ub = penalty*K.relu(K.square(y_pred - Umin))
                return K.sum(mse + lb + ub, axis=-1)
            loss = bounded_loss
            
        self.model.compile(loss=loss, optimizer=optimizer)
        self.loss = loss

        super(KerasQ, self).__init__(**kwargs)

    def call(self, state):
        return self.model.predict_on_batch(state)

    def update(self, update, state, n=None):
        if n:
            for _ in xrange(n):
                loss = self.model.train_on_batch(state, update)
            return loss 
        else:
            return self.model.train_on_batch(state, update)

    def save(self, s_dir, name='Qmodel', **kwargs):
        self.model.save('{}/{}.h5'.format(s_dir,name))

    def load(self, s_dir, name='Qmodel', custom_objects=None, **kwargs):
        self.model = load_model('{}/{}.h5'.format(s_dir,name), custom_objects=custom_objects)
        self.model.compile(**kwargs)

class KNNQ(ValueFunction):
    def __init__(self, S, A, n_neighbors=5, weights='uniform', algorithm='auto', metric='minkowski', memory_fit=100, memory_size=100, **kwargs):
        #assert self.lr_mode == 'constant', 'KNNQ is only compatible with constant learning rates.'
        self.S = S
        self.A = A
        self.states = deque([])
        self.targets = deque([])
        self.memory_fit = memory_fit
        self.memory_size = memory_size
        self.count = 0

        self.neigh = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, metric=metric)

        super(KNNQ, self).__init__(**kwargs)
        self.update_mode = 'set'

    def call(self, state, *args):
        if len(args) > 0:
            action = args[0]
        else:
            action = None
        if self.count > self.memory_fit+1:
            if action != None:
                return self.neigh.predict(state.reshape(1,-1))[0,action]
            else:
                return self.neigh.predict(state.reshape(1,-1))
        else:
            if action != None:
                return np.random.rand(self.A.n)[action]
            else:
                return np.random.rand(self.A.n)

    def update(self, update, state, n=None):
        # calculate target
        target = self.lr(n) * update
        # control memory growth
        if len(self.states) >= self.memory_size:
            #idx = np.random.choice(len(self.states))
            #self.states.pop(idx)
            #self.targets.pop(idx)
            #self.states.insert(idx, state)
            #self.targets.insert(idx, target)
            self.states.popleft()
            self.targets.popleft()
            self.states.append(state)
            self.targets.append(target)
        else:
            self.states.append(state)
            self.targets.append(target)
        if self.count % self.memory_fit == 0 and self.count != 0:
            self.neigh.fit(np.array(self.states), np.array(self.targets))
        self.count += 1

class TableQ(ValueFunction):
    def __init__(self, S, A, **kwargs):
        self.S = S
        self.A = A

        assert type(S) == type(A) == gym.spaces.Discrete, 'TableQ is only compatible with Discrete spaces.'
        s_dim = get_space_dim(S)
        a_dim = get_space_dim(A)
        self.table = np.zeros((s_dim, a_dim))

        super(TableQ, self).__init__(**kwargs)

    def call(self, state, *args):
        if len(args) > 0:
            action = args[0]
            assert state < self.S.n and state >= 0
            assert action < self.A.n and action >= 0
            return self.table[state,action]
        else:
            assert state < self.S.n and state >= 0
            return self.table[state]

    def update(self, update, state, n=None):
        self.table[state] += self.lr(n) * update #update is an a_dim vector with zeros on all but the target action
