import numpy as np
import gym
from keras.models import Model, Sequential, load_model
from keras.layers import Convolution2D, Dense, Flatten
from keras.preprocessing import image
import pickle as pkl

from time import sleep
#import cv2
from PIL import Image

# Very Important: it defaults to tensor flow
import keras.backend as K
print K.image_dim_ordering()
#K.set_image_dim_ordering('th')
#print K.image_dim_ordering()

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
    def __init__(self, S, A):
        self.S
        self.A

    def __call__(self, state, *args):
        return self.call(state, *args)

    def call(self, state, *args):
        raise NotImplementedError

    def update(self, update):
        raise NotImplementedError

class QTable(ValueFunction):
    def __init__(self, S, A):
        self.S = S
        self.A = A

        assert type(S) == type(A) == gym.spaces.Discrete
        s_dim = get_space_dim(S)
        a_dim = get_space_dim(A)
        self.table = np.zeros((s_dim, a_dim))

    def call(self, state=None, action=None):
        if state == None:
            if action == None:
                return self.table
            else:
                assert action < self.A.n and action >= 0
                return self.table[:,action]
        else:
            if action == None:
                assert state < self.S.n and state >= 0
                return self.table[state]
            else:
                assert state < self.S.n and state >= 0
                assert action < self.A.n and action >= 0
                return self.table[state,action]

    def update(self, update, state, action):
        self.table[state, action] = update

class KerasQ(ValueFunction):
    def __init__(self, S, A, model):
        self.S = S
        self.A = A
        self.model = model

        self.loss = 0

    def call(self, state):
        return self.model.predict(state)

    def update(self, states, targets):
        self.loss = self.model.train_on_batch(states, targets)

    def explicit_update(self, weights):
        self.model.set_weights(weights)

    def get_loss(self):
        return self.loss

class Policy(object):
    def __init__(self, S, A):
        self.S = S
        self.A = A

        self.s_dim = get_space_dim(S)
        self.a_dim = get_space_dim(A)

    def __call__(self, state, **kwargs):
        return self.call(state, **kwargs)

    def call(self, state, **kwargs):
        return self.A.sample()

    def update(self):
        raise NotImplementedError
    

class Agent(object):
    def __init__(self, name=None, lr_mode='constant'):
        #self.S = S
        #self.A = S
        #self.policy = policy
        self.name = name
        self.lr_mode = lr_mode

        #self.state_dim = get_space_dim(S)
        #self.action_dim = get_space_dim(A)      

    def observe(self, s_next, r, done):
        self.state = s_next
    
    def act(self):
        action = self.policy(self.state)
        self.action = action
        return self.action   

class EpsilonGreedy(Policy):
    def __init__(self, S, A, Qfunction=QTable, epsilon=0.1):
        self.S = S
        self.A = A
        self._epsilon = epsilon

        self.Qfunction = Qfunction

    def call(self, state, epsilon=None):
        if epsilon:
            self.epsilon = epsilon
        rand = np.random.random()
        if rand > self.epsilon:
            return self.Qfunction(state).argmax()
        else:
            return self.A.sample()

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        self._epsilon = value

class KerasActor(Policy):
    def __init__(self, S, A, model):
        self.S = S
        self.A = A
        self.model = model

        self.loss = 0

    def call(self, state):
        return self.model.predict(state)

    def grad_log(self, state):
        raise NotImplementedError

class ReplayMemory(object):
    def __init__(self, memory_size, state_shape, dtype=np.uint8):
        self.memory_size = memory_size

        if type(state_shape) == int:
            states_shape = (memory_size, state_shape)
        else:
            states_shape = (memory_size,) + tuple(state_shape)
        self.states = np.zeros(states_shape, dtype=dtype)
        self.actions = np.zeros((memory_size,), dtype=np.uint8)
        self.next_states = np.zeros(states_shape, dtype=dtype)
        self.rewards = np.zeros((memory_size,), dtype=dtype)

        self.idx = 0 #memory location
        self.full = False

    def add(self, transition):
        state, action, next_state, reward = transition        
        #for x in transition:
        #    print x.shape
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.next_states[self.idx] = next_state
        self.rewards[self.idx] = reward

        # Update the memory location for the next add
        if self.idx == (self.memory_size - 1):
            self.idx = 0
            self.full = True
        else:
            self.idx += 1 
    
    def get_minibatch(self, batch_size):
        if self.full:
            idxs = np.random.choice(self.memory_size, size=batch_size, replace=False)
        else:
            idxs = np.random.choice(self.idx, size=min(batch_size,self.idx), replace=False)

        states = self.states[idxs]
        actions = self.actions[idxs]
        next_states = self.next_states[idxs]
        rewards = self.rewards[idxs]
        return states, actions, next_states, rewards

    def save(self, dir='.'):
        np.save('{}/states.npy'.format(dir), self.states)
        np.save('{}/next_states.npy'.format(dir), self.next_states)
        np.save('{}/actions.npy'.format(dir), self.actions)
        np.save('{}/rewards.npy'.format(dir), self.rewards)

    def load(self, f_states, f_next_states, f_actions, f_rewards):
        # My hope is that deleting the memory allows me to load the memory
        del self.states
        del self.next_states
        del self.actions
        del self.rewards
        self.states = np.load(f_states)
        self.next_states = np.load(f_next_states)
        self.actions = np.load(f_actions)
        self.rewards = np.load(f_rewards)

class KerasDQN(Agent):
    def __init__(self, S, A, model=None, policy=EpsilonGreedy, gamma=0.99, memory_size=10000, update_freq=10000, batch_size=32, **kwargs):
        self.S = S
        self.A = A
        assert type(A) == gym.spaces.Discrete

        self.gamma = gamma
        self.memory_size = memory_size
        self.update_freq = update_freq
        self.batch_size = batch_size

        super(KerasDQN, self).__init__(**kwargs)

        if type(S) == gym.spaces.Box:
            state_shape = S.shape
        elif type(S) == gym.spaces.Discrete:
            state_shape = S.n
        else:
            raise TypeError

        if not self.name:
            self.name = 'DQN'

        # Use provided model else use atari dqn model
        if model != None:
            self.model = model
        else:
            if K.image_dim_ordering() == 'th':
                state_shape = (4,84,84)
            else:
                state_shape = (84,84,4)
            # default DQN as for atari
            model = Sequential()
            model.add(Convolution2D(32, 8, 8, subsample=(4,4), input_shape=state_shape, activation='relu'))
            model.add(Convolution2D(64, 4, 4, subsample=(2,2), activation='relu'))
            model.add(Convolution2D(64, 3, 3, subsample=(1,1), activation='relu'))
            model.add(Flatten())
            model.add(Dense(512, activation='relu'))
            model.add(Dense(A.n))

            model.compile(loss='mse', optimizer='rmsprop')
            self.model = model

        self.Qfunction = KerasQ(S, A, self.model)
        self.policy = policy(S, A, self.Qfunction)

        self.target_model = Sequential.from_config(self.model.get_config())
        self.target_model.set_weights(self.model.get_weights())
        self.target_Qfunction = KerasQ(S, A, self.target_model)

        self.replay_memory = ReplayMemory(memory_size, state_shape, dtype=np.uint8)

           
        # only a single frame 
        self.prev_state = S.sample()

        # initial states and action
        states = [self.preprocess_input(S.sample()) for i in xrange(4)]
        if K.image_dim_ordering() == 'th':
            self.state = np.concatenate(states, axis=1)
        else:
            self.state = np.concatenate(states, axis=-1)
        self.state.fill(0)
        self.action = np.asarray(A.sample())
        self.action.fill(0)

        self.time = 0
        self.episode = 0

        self.frame = 0
        self.frame_count = 0
        self.m = 4
        self.s_next_frames = []
        self.replay_start_size = 50000
        self.action_repeat = 4
        self.action_count = 0
        self.update_freq = 4

    def observe(self, s_next, r, done):
        self.frame_count += 1
        
        # collect m frames for input
        if self.frame < self.m:
            self.s_next_frames.append(self.preprocess_input(s_next))
            self.prev_state = s_next
            self.frame += 1
            return self.Qfunction.get_loss() 

        if K.image_dim_ordering() == 'th':    
            s_next = np.concatenate(self.s_next_frames, axis=1)
        else:
            s_next = np.concatenate(self.s_next_frames, axis=-1)
        self.frame = 0
        self.s_next_frames = []
        
        #s_next = s_next[np.newaxis,:] # turn vector into matrix

        self.done = done
        self.time_management(done)

        transition = (self.state, self.action, s_next, r)
        self.replay_memory.add(transition)
    
        if self.frame_count >= self.replay_start_size and self.action_count % self.update_freq == 0:
            minibatch = self.replay_memory.get_minibatch(self.batch_size)
            self.batch_updates(minibatch)

        self.state = s_next

        return self.Qfunction.get_loss()

    def act(self, **kwargs):
        # repeat action for action_repeat times
        if self.frame_count % self.action_repeat == 0:
            # count how many actions you have selected
            self.action_count += 1
            # for less than replay_start_size apply a random policy
            if self.frame_count < self.replay_start_size:
                self.action = self.A.sample()
            else:
                self.action = self.policy(self.state, **kwargs)
        return self.action

    def batch_updates(self, minibatch):
        #for x in minibatch:
        #    print x.shape
        states, actions, next_states, rewards = minibatch
        targets = np.zeros((states.shape[0], self.A.n))
        #print states.shape
        targets = self.Qfunction(states)

        if self.done:
            targets[:,actions] = rewards[:,np.newaxis]
        else:
            Qs = self.target_Qfunction(next_states)
            #print targets.shape
            #print actions
            #print targets[:,actions].shape
            targets[:,actions] = (rewards + self.gamma * Qs.argmax(1))[:,np.newaxis]
        self.Qfunction.update(states, targets)

    def preprocess_input(self, state):
        # I need to process the state with the previous state
        # I take the max pixel value of the two frames
        state = np.maximum(state, self.prev_state)
        state = np.asarray(state, dtype=np.float32)
        # convert rgb image to yuv image
        #yuv = cv2.cvtColor(state,cv2.COLOR_RGB2YUV)
        yCbCr = Image.fromarray(state, 'YCbCr')
        # extract the y channel
        #y, u, v = cv2.split(yuv)
        y, Cb, Cr = yCbCr.split()
        # rescale to 84x84
        #y_res = cv2.resize(y, (84,84), interpolation=cv2.INTER_AREA)
        y_res = y.resize((84,84))
        y_res = image.img_to_array(y_res)
        y_res = y_res[np.newaxis,:,:]    
        return y_res

    def time_management(self, done):
        self.time += 1
        if done:
            self.episode += 1 
        if self.time % self.update_freq == 0:
            self.target_Qfunction.explicit_update(self.model.get_weights())

    def save(self, dir='.'):
        self.model.save('{}/model.h5'.format(dir))
        self.target_model.save('{}/target_model.h5'.format(dir))
        self.replay_memory.save(dir=dir)

        d = {'gamma': self.gamma, 'update_freq': self.update_freq, 'batch_size': self.batch_size,
             'time': self.time, 'episode': self.episode, 'frame': self.frame, 'frame_count': self.frame_count,
             'action_count': self.action_count}
        pkl.dump(d, open('{}/config.pkl'.format(dir), 'w'))
             

    def load(self, f_model, f_target_model, f_states, f_next_states, f_actions, f_rewards, f_config):
        self.model = load_model(f_model)
        self.model.compile(loss='mse', optimizer='rmsprop')
        self.target_model = load_model(f_target_model)
        self.target_model.compile(loss='mse', optimizer='rmsprop')

        d = pkl.load(open(f_config, 'r'))
        self.gamma = d['gamma']
        self.update_freq = d['update_freq']
        self.batch_size = d['batch_size']
        self.time = d['time']
        self.episode = d['episode']
        #self.frame = d['frame']
        self.frame_count = d['frame_count']
        self.action_count = d['action_count']

        self.Qfunction.model = self.model
        self.target_Qfunction.model = self.target_model
        self.replay_memory.load(f_states, f_next_states, f_actions, f_rewards)
        self.policy.model = self.model

class QLearning(Agent):
    def __init__(self, S, A, policy=EpsilonGreedy, learning_rate=1e-3, gamma=0.99, **kwargs):         
        self.S = S
        self.A = A

        self.gamma = gamma

        self.Q = QTable(S, A)
        self.policy = EpsilonGreedy(S, A, Qfunction=self.Q)

        super(QLearning, self).__init__(**kwargs)
        if not self.name:
            self.name = 'QLearning'

        if self.lr_mode == 'constant':
            self.learning_rate = learning_rate
        else:
            self.n = np.zeros_like(self.Q.table) #initialize as zero matrix with shape (s_dim, a_dim)
            self.learning_rate = learning_rate

        self.state = 0
        self.action = 0

    def observe(self, s_next, r, done):
        if done:
            Q_est = r
        else:
            Q_est = r + self.gamma*self.Q(s_next).max()
        update = Q_est - self.Q(self.state, self.action)

        if self.lr_mode == 'constant':
            updateQ = self.Q(self.state, self.action) + self.learning_rate*update
        else:
            self.n[self.state, self.action] += 1 # count updates
            n = self.n[self.state, self.action]
            updateQ = self.Q(self.state, self.action) + self.learning_rate(n)*update
        self.Q.update(updateQ, self.state, self.action)

        self.state = s_next
        return update

    def act(self, **kwargs):
        self.action = self.policy(self.state, **kwargs)
        return self.action
     
class DoubleQLearning(Agent):
    def __init__(self, S, A, policy=EpsilonGreedy, learning_rate=1e-3, gamma=0.99, **kwargs):         
        self.S = S
        self.A = A

        self.gamma = gamma

        super(DoubleQLearning, self).__init__(**kwargs)

        if not self.name:
            self.name = 'DoubleQ'
        
        self.Q_A = QTable(S, A)
        self.Q_B = QTable(S, A)

        if self.lr_mode == 'constant':
            self.learning_rate = learning_rate
        else:
            self.nA = np.zeros_like(self.Q_A.table) #initialize as zero matrix with shape (s_dim, a_dim)
            self.nB = np.zeros_like(self.Q_B.table) #initialize as zero matrix with shape (s_dim, a_dim)
            self.learning_rate = learning_rate

        def doubleQ(state):
            ave_Q = (self.Q_A(state) + self.Q_B(state)) / 2
            return ave_Q
        
        self.policy = EpsilonGreedy(S, A, Qfunction=doubleQ)

        self.state = 0
        self.action = 0

    def observe(self, s_next, r, done):
        # randomly choose either update A or B
        rand = np.random.random()
        if rand > 0.5:
            if done:
                Q_est = r
            else:
                a_star = self.Q_A(s_next).argmax()
                Q_est = r + self.gamma*self.Q_B(s_next, a_star)
            update = Q_est - self.Q_A(self.state, self.action)
            if self.lr_mode == 'constant':
                updateA = self.Q_A(self.state, self.action) + self.learning_rate*update
            else:
                self.nA[self.state, self.action] += 1 # count updates
                n = self.nA[self.state, self.action]
                updateA = self.Q_A(self.state, self.action) + self.learning_rate(n)*update
            
            self.Q_A.update(updateA, self.state, self.action)
        else:
            if done:
                Q_est = r
            else:
                b_star = self.Q_B(s_next).argmax()
                Q_est = r + self.gamma*self.Q_A(s_next, b_star)
            update = Q_est - self.Q_B(self.state, self.action)
            if self.lr_mode == 'constant':
                updateB = self.Q_B(self.state, self.action) + self.learning_rate*update
            else:
                self.nB[self.state, self.action] += 1 # count updates
                n = self.nB[self.state, self.action]
                updateB = self.Q_B(self.state, self.action) + self.learning_rate(n)*update
            self.Q_B.update(updateB, self.state, self.action)

        self.state = s_next
        return update

    def act(self, **kwargs):
        self.action = self.policy(self.state, **kwargs)
        return self.action

def softmax(x):
    y = np.exp(x)
    return y / y.sum(-1)

class LinearPolicy(Policy):
    def __init__(self, S, A, degree=1):
        super(LinearPolicy, self).__init__(S, A)

        self.degree = degree
        self.basis_fn = [lambda x: np.power(x,i) for i in xrange(self.degree + 1)]
        self.param_dim = len(self.basis_fn) * self.s_dim * self.a_dim
        
    def call(self, state, params, ):
        # flatten the state so that it can work with the parameters
        if len(state.shape) > 1:
            state = np.ravel(state)

        # create extended state using basis functions
        full_state_list = [fn(state) for fn in self.basis_fn]
        full_state = np.concatenate(full_state_list)
        
        # create matrix of parameters
        params = params.reshape(self.a_dim, full_state.shape[0])
        
        predictor = np.dot(params,full_state)
        return predictor
        

class CrossEntropy(Agent):
    def __init__(self, S, A, policy=LinearPolicy, n_sample=100, top_p=0.2, init_mean=None, init_std=None, **kwargs):
        self.S = S
        self.A = A

        self.policy = policy(S, A, degree=1)

        self.param_dim = self.policy.param_dim

        self.n_sample = n_sample
        self.top_p = top_p
        self.n_elite = int(top_p * n_sample)

        if init_mean == None:
            self.mean = np.random.uniform(low=-1,high=1,size=self.param_dim)
            
        if init_std == None:
            self.std = np.random.uniform(low=0,high=1,size=self.param_dim) 

        super(CrossEntropy, self).__init__(**kwargs)

        if not self.name:
            self.name = 'CrossEntropy'

        self.i = 0 #sample counter varies between 0 and n_sample-1
        self.R = np.zeros(self.n_sample) # the collection of rewards

    def observe(self, s_next, r, done):
        # collect n parameter samples
        if self.i == 0:
            self.params = np.random.multivariate_normal(self.mean, np.diag(self.std), self.n_sample)

        # evaluate the parameters
        self.R[self.i] += r

        # select the elite set if all params have been evaluated
        if self.i >= self.n_sample-1:
            print 'Selecting Elite...'
            sleep(0.1)
            indices = np.argsort(self.R)
            self.elite = self.params[indices][-self.n_elite:]
            self.mean = self.elite.mean(0)
            self.std =  self.elite.std(0)
            self.i = 0
        
        if done:
            self.i += 1

        self.state = s_next

        return 0
   
    def act(self, **kwargs):
        predictor = self.policy(self.state, params=self.params[self.i])

        if type(self.A) == gym.spaces.Discrete:
            self.action = softmax(predictor).argmax()
        else:
            self.action = predictor
        return self.action
             
