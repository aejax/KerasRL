import numpy as np
import gym
from keras.models import Model, Sequential

from time import sleep

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
    def __init__(self, name=None):
        #self.S = S
        #self.A = S
        #self.policy = policy
        self.name = name

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
    def __init__(self, memory_size, state_shape):
        self.memory_size = memory_size

        if type(state_shape) == int:
            states_shape = (memory_size, state_shape)
        else:
            states_shape = (memory_size,) + tuple(state_shape)
        self.states = np.zeros(states_shape)
        self.actions = np.zeros((memory_size,), dtype='int32')
        self.next_states = np.zeros(states_shape)
        self.rewards = np.zeros((memory_size,))

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

class KerasDQN(Agent):
    def __init__(self, S, A, model, policy=EpsilonGreedy, gamma=0.99, memory_size=1000, update_freq=100, batch_size=32, **kwargs):
        self.S = S
        self.A = A
        assert type(A) == gym.spaces.Discrete

        self.gamma = gamma
        self.memory_size = memory_size
        self.update_freq = update_freq
        self.batch_size = batch_size

        super(KerasDQN, self).__init__(**kwargs)

        if not self.name:
            self.name = 'DQN'

        self.model = model
        self.Qfunction = KerasQ(S, A, model)
        self.policy = policy(S, A, self.Qfunction)

        self.target_model = Sequential.from_config(self.model.get_config())
        self.target_model.set_weights(self.model.get_weights())
        self.target_Qfunction = KerasQ(S, A, self.target_model)

        if type(S) == gym.spaces.Box:
            state_shape = S.shape
        elif type(S) == gym.spaces.Discrete:
            state_shape = S.n
        else:
            raise TypeError

        self.replay_memory = ReplayMemory(memory_size, state_shape)

        self.state = S.sample()
        self.state.fill(0)
        self.action = np.asarray(A.sample())
        self.action.fill(0)

        self.time = 0
        self.episode = 0

    def observe(self, s_next, r, done):
        s_next = s_next[np.newaxis,:] # turn vector into matrix

        self.done = done
        self.time_management(done)

        transition = (self.state, self.action, s_next, r)
        self.replay_memory.add(transition)
        minibatch = self.replay_memory.get_minibatch(self.batch_size)
        
        self.batch_updates(minibatch)

        self.state = s_next

        return self.Qfunction.get_loss()

    def act(self, **kwargs):
        self.action = self.policy(self.state, **kwargs)
        return self.action

    def batch_updates(self, minibatch):
        states, actions, next_states, rewards = minibatch
        targets = np.zeros((states.shape[0], self.A.n))

        targets = self.Qfunction(states)

        if self.done:
            targets[actions] = rewards[:,np.newaxis]
        else:
            Qs = self.target_Qfunction(next_states)
            targets[actions] = (rewards + self.gamma * Qs.argmax(1))[:,np.newaxis]
        self.Qfunction.update(states, targets)

    def time_management(self, done):
        self.time += 1
        if done:
            self.episode += 1 
        if self.time % self.update_freq == 0:
            self.target_Qfunction.explicit_update(self.model.get_weights())

class QLearning(Agent):
    def __init__(self, S, A, policy=EpsilonGreedy, learning_rate=1e-3, gamma=0.99, **kwargs):         
        self.S = S
        self.A = A

        self.learning_rate = learning_rate
        self.gamma = gamma

        super(QLearning, self).__init__(**kwargs)

        if not self.name:
            self.name = 'QLearning'
        
        self.Q = QTable(S, A)
        
        self.policy = EpsilonGreedy(S, A, Qfunction=self.Q)

        self.state = 0
        self.action = 0

    def observe(self, s_next, r, done):
        if done:
            Q_est = r
        else:
            Q_est = r + self.gamma*self.Q(s_next).max()
        update = Q_est - self.Q(self.state, self.action)
        updateQ = self.Q(self.state, self.action) + self.learning_rate*update
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

        self.learning_rate = learning_rate
        self.gamma = gamma

        super(DoubleQLearning, self).__init__(**kwargs)

        if not self.name:
            self.name = 'DoubleQ'
        
        self.Q_A = QTable(S, A)
        self.Q_B = QTable(S, A)

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
            updateA = self.Q_A(self.state, self.action) + self.learning_rate*update
            self.Q_A.update(updateA, self.state, self.action)
        else:
            if done:
                Q_est = r
            else:
                b_star = self.Q_B(s_next).argmax()
                Q_est = r + self.gamma*self.Q_A(s_next, b_star)
            update = Q_est - self.Q_B(self.state, self.action)
            updateB = self.Q_B(self.state, self.action) + self.learning_rate*update
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
             
