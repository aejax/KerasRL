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

    def load(self, s_dir, name='Qmodel', custom_objects={}, **kwargs):
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
        super(TableQ, self).__init__(**kwargs)
        self.S = S
        self.A = A

        assert type(S) == type(A) == gym.spaces.Discrete, 'TableQ is only compatible with Discrete spaces.'
        s_dim = get_space_dim(S)
        a_dim = get_space_dim(A)
        self.table = np.zeros((s_dim, a_dim))

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

def add_Q(a,b):
    o = np.zeros_like(a)
    for i in range(a.shape[0]):
        x = a[i]
        y = b[i]
        if np.isinf(y):
            o[i] = x
        elif np.isinf(x):
            o[i] = y
        else:
            o[i] =  x+y
    return o

class TableQ2(ValueFunction):
    def __init__(self, S, A, maxlen=1000, mode=None, embedding_dim=1, **kwargs):
        super(TableQ2, self).__init__(**kwargs)
        self.S = S
        self.A = A

        if mode == None:
            if type(S) == type(A) == gym.spaces.Discrete:
                self.mode = 'array'
            elif type(A) == gym.spaces.Discrete:
                self.mode = 'dictionary'
            else:
                pass
        self.mode = mode
        self.maxlen = maxlen
        self.embedding_dim = embedding_dim

        if self.mode == 'array':
            s_dim = get_space_dim(S)
            a_dim = get_space_dim(A)
            self.table = np.zeros((s_dim, a_dim))
            self.maxlen = s_dim
        elif self.mode == 'dictionary':
            self.table = {0: np.zeros(self.A.n)}
        elif self.mode == 'tables':
            self.k = 4
            self.neigh = KNeighborsRegressor(n_neighbors=self.k)
            self.states = np.zeros((self.maxlen,self.embedding_dim))
            self.values = np.zeros((self.maxlen, self.A.n))
            self.recency= np.zeros((self.maxlen,))
            self.i = 0
        elif self.mode == 'action_tables':
            #self.states = []
            #self.recency= []
            self.k = 4
            self.action_tables = [ [[],[], KNeighborsRegressor(n_neighbors=self.k), []]
                                   for _ in xrange(self.A.n)]
            for states, values, neigh, recency in self.action_tables:
                for _ in xrange(self.k):
                    states.append(np.ones(self.embedding_dim))
                    values.append(1)
                    recency.append(0)
                neigh.fit(np.array(states), np.array(values))
        else:
            raise NotImplementedError, 'Sorry, TableQ only supports three modes.'

    def call(self, state, *args):
        if self.mode == 'array':
            if len(args) > 0:
                action = args[0]
                return self.table[state,action]
            else:
                return self.table[state]        
        elif self.mode == 'dictionary':
            if state not in self.table:
                k = 4
                states = self.table.keys()
                indices = np.argsort(np.abs(np.array(states) - state))
                closes = [states[ind] for ind in indices[:k]] 
                Q = [self.table[close] for close in closes]
                Q = reduce(add_Q, Q)
                Q /= k
            else:
                Q = self.table[state]
            if len(args) > 0:
                action = args[0]
                return Q[action]
            else:
                return Q
        elif self.mode == 'tables':
            if len(np.where(self.states == state)[0]) == 0:
                Q = self.neigh.predict(state)[0]
                #print 'Q value: ', Q
                #k = 1
                #indices = np.argsort(np.abs(self.states - state)) 
                #Q = self.values[indices[:k]]
                #Q = reduce(add_Q, Q)
                #Q /= k
            else:
                Q = self.values[(np.where(self.states == state)[0][0],)]
            if len(args) > 0:
                action = args[0]
                return Q[action]
            else:
                return Q
        elif self.mode == 'action_tables':
            #print 'Calling Action Tables...'
            if len(args) > 0:
                action = args[0]
                states, values, neigh, recency = self.action_tables[action]
                if state in states:
                    i = states.index(state)
                    return values[i]
                else:
                    return neigh.predict(state)[0]
            else:
                Q = []
                for states, values, neigh, recency in self.action_tables:
                    #if state in states:
                    t = [np.array_equal(state,x) for x in states]
                    if reduce(lambda x,y: x or y, t):
                        #i = states.index(state)
                        i = t.index(True)
                        Q.append(values[i])
                    else:
                        if type(state) != float and state.ndim == 1:
                            state = state[np.newaxis,:]
                        q = neigh.predict(state)[0]
                        if type(q) == np.ndarray:
                            q = q[0]
                        Q.append(q)
                Qa = np.array(Q)
                assert Qa.ndim == 1, q
                return Qa
        else:
            pass

    def update(self, update, state, n=None):
        if self.mode == 'array':
            self.table[state] += self.lr(n) * update #update is an a_dim vector with zeros on all but the target action
        elif self.mode == 'dictionary':
            if len(self.table) >= self.maxlen:
                key = self.table.keys()[0]
                del self.table[key] #delete an item
                self.table[state] = update
            else:
                self.table[state] = update
        elif self.mode == 'tables':
            if self.i >= self.maxlen-1:
                i = np.argmax(self.recency)
                self.recency += 1
                self.states[i] = state
                self.values[i] = update
                self.recency[i] = 0
            else:
                if len(np.where(self.states == state)[0]) == 0:
                    self.i += 1
                    self.recency += 1
                    self.states[self.i] = state
                    self.values[self.i] = update
                    self.recency[self.i] = 0
                else:
                    self.recency += 1
                    self.values[(np.where(self.states == state)[0][0],)] = update
                    self.recency[(np.where(self.states == state)[0][0],)] = 0
        elif self.mode == 'action_tables':
            for i in xrange(self.A.n):
                if update[i] != 0:
                    states, values, neigh, recency = self.action_tables[i]
                    #if state in states:
                    t = [np.array_equal(state,x) for x in states]
                    if reduce(lambda x,y: x or y, t):
                        map(lambda x: x+1,recency)
                        #j = states.index(state)
                        j = t.index(True)
                        values[j] = update[i]
                        recency[j] = 0
                    else:
                        if len(states) > self.maxlen:
                            map(lambda x: x+1,recency)
                            old = max(recency)
                            j = recency.index(old)
                            states[j] = state
                            values[j] = update[i]
                            recency[j] = 0
                        else:
                            states.append(state)
                            values.append(update[i])
                            recency.append(0)
                else:
                    pass             
        else:
            raise NotImplementedError, 'TableQ supports these four modes: array, dictionary, tables, and action_tables.'

    def fit(self):
        if self.mode == 'tables':
            self.neigh.fit(self.states, self.values)
        elif self.mode == 'action_tables':
            for states, values, neigh, recency in self.action_tables:
                s = self._list_to_sklearn(states)
                v = self._list_to_sklearn(values)
                neigh.fit(s, v)
    
    def _list_to_sklearn(self, l):
        a = np.array(l)
        if a.ndim == 1:
            a = a[:,np.newaxis]
        return a
    
    def contains(self, state, *args):
        if self.mode == 'dictionary':
            if state in self.table:
                Q = self.table[state]
            else:
                return False

            if len(args) > 0:
                action = args[0]
                return not np.isinf(Q[action])
            else:
                return True

        elif self.mode == 'array':
            if len(args) > 0:
                action = args[0]
                return not np.isinf(self.table[state,action])
            else:
                return True
        elif self.mode == 'tables':
            if len(np.where(self.states == state)[0]) != 0:
                Q = self.values[(np.where(self.states == state)[0][0],)]
            else:
                return False
            
            if len(args) > 0:
                action = args[0]
                #return not np.isinf(Q[action])
                return Q[action] != 0
            else:
                return True
        elif self.mode == 'action_tables':
            if len(args) > 0:
                action = args[0]
                states, values, neigh, recency = self.action_tables[action]
                #if state in states:
                t = [np.array_equal(state,x) for x in states]
                if reduce(lambda x,y: x or y, t):
                    return True
                else:
                    return False
            else:
                for states, values, neigh, recency in self.action_tables:
                    #if state in states:
                    t = [np.array_equal(state,x) for x in states]
                    if reduce(lambda x,y: x or y, t):
                       return True
                return False
        else:
            pass

    def save(self, s_dir):
        pkl.dump(self.table, open('{}/{}.pkl'.format(s_dir,self.name)))

    def load(self, l_dir):
        self.table = pkl.load(open('{}/{}.pkl'.format(l_dir,self.name)))
