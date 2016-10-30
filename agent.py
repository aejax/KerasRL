from value import *
from collections import deque
import copy
from PIL import Image
import keras.backend as K
import timeit
import cPickle as pkl

def islambda(v):
    LAMBDA = lambda:0
    return isinstance(v, type(LAMBDA)) and v.__name__ == LAMBDA.__name__

def softmax(x):
    y = np.exp(x)
    return y / y.sum(-1)

def epsilon_greedy(A, epsilon=0.1):
    def f(action, **kwargs):
        rand = np.random.random()
        if rand > epsilon:
            return action
        else:
            return A.sample()
    return f

def SAepsilon_greedy(A, epsilon=0.1, final=1000, schedule=None):
    if schedule == None:
        schedule = lambda n, f: (f-n)/f
    def f(action, n=0):
        if n <= final: 
            eps = max(epsilon, schedule(float(n),final))
        else:
            eps = epsilon
        rand = np.random.random()
        if rand > eps:
            return action
        else:
            return A.sample()
    return f
    

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

class Policy(object):
    def __init__(self, A=None, name=None, randomness=None):
        self.A = A
        if name == None:
            self.name = self.__class__.__name__
        else:
            self.name = name
        if randomness == None:
            self.randomness = lambda action, **kwargs: action
        else:
            self.randomness = randomness
            
    def __call__(self, state, **kwargs):
        return self.randomness(self.call(state, **kwargs), **kwargs)

    def call(self, state, *args, **kwargs):
        return self.A.sample()

    def update(self, update, *args, **kwargs):
        pass

    def save(self, filename, **kwargs):
        d = {'A': self.A, 'name': self.name}
        pkl.dump(d, open(filename, 'w'))

    def load(self, filename, **kwargs):
        d = pkl.load(open(filename, 'r'))
        for key, value in d.iteritems():
            if hasattr(self, key):
                self.__dict__[key] = value

class MaxQ(Policy):
    def __init__(self, Qfunction, **kwargs):
        self.Qfunction = Qfunction

        super(MaxQ, self).__init__(**kwargs)

    def call(self, state, **kwargs):
        return self.Qfunction(state).argmax()

    def update(self, update, *args, **kwargs):
        return self.Qfunction.update(update, *args, **kwargs)

    def save(self, s_dir, **kwargs):
        self.Qfunction.save(s_dir, **kwargs)

    def load(self, l_dir, **kwargs):
        self.Qfunction.load(l_dir, **kwargs)
        

class ConsensusQ(Policy):
    def __init__(self, Qfunctions, **kwargs):
        self.Qfunctions = Qfunctions

        super(ConsensusQ, self).__init__(**kwargs)

    def call(self, state, **kwargs):
        consensus = reduce(lambda x, y: x(state) + y(state), self.Qfunctions)
        consensus /= len(self.Qfunctions)
        action = consensus.argmax()
        return self.randomness(action, **kwargs)

    def update(self, update, idx=0, *args, **kwargs):
        self.Qfunctions[idx].update(update, *args, **kwargs)

    def save(self, s_dir, **kwargs):
        for Q in self.Qfunctions:
            Q.save(s_dir, **kwargs)

    def load(self, l_dir, **kwargs):
        for Q in self.Qfunctions:
            Q.load(l_dir, **kwargs)

class Linear(Policy):
    def __init__(self, S=None, basis_fns=None, degree=1, **kwargs):

        super(Linear, self).__init__(**kwargs)

        self.S = S
        assert self.A and self.S, 'Linear Policy requires both state and action spaces'
        self.a_dim = get_space_dim(A)
        self.s_dim = get_space_dim(S)

        self.degree = degree

        # default basis functions are polynomials
        if basis_fns == None:
            self.basis_fns = [lambda x: np.power(x,i) for i in xrange(self.degree + 1)]
        else:
            self.basis_fns = basis_fns
        self.param_dim = len(self.basis_fns) * self.s_dim * self.a_dim

    def call(self, state, params):
        # flatten the state so that it can work with the parameters
        if len(state.shape) > 1:
            state = np.ravel(state)

        # create extended state using basis functions
        full_state_list = [fn(state) for fn in self.basis_fns]
        full_state = np.concatenate(full_state_list, axis=0)
        
        # create matrix of parameters
        params = params.reshape(self.a_dim, full_state.shape[0])
        
        predictor = np.dot(params,full_state)
        return predictor
 
class Agent(object):
    def __init__(self, policy=None, name=None, random_start=0):
        self.policy = policy
        self.random_start = random_start
        if name == None:
            self.name = self.__class__.__name__
        else:
            self.name = name

        self.state = None
        self.action = None
    def observe(self, s_next, r, done, info=None):
        self.state = s_next

    def act(self, **kwargs):
        self.action = self.policy(self.state, **kwargs)
        return self.action

    def save(self, s_dir, **kwargs):
        self.policy.save(s_dir, **kwargs)

    def load(self, l_dir, **kwargs):
        self.policy.load(l_dir, **kwargs)

class QLearning(Agent):
    def __init__(self, S, A, Q=None, gamma=0.99, lr=1e-3, **kwargs):
        self.S = S
        self.A = A
        if Q == None:
            self.Q = TableQ(S, A, lr=lr)
        else:
            self.Q = Q
        self.gamma = gamma

        super(QLearning, self).__init__(**kwargs)

        if self.policy == None:
            self.policy = MaxQ(self.Q)
        self.Q = self.policy.Qfunction

        self.state = self.S.sample()
        self.action = self.A.sample()

    def observe(self, s_next, r, done, n):
        update = np.zeros(self.A.n)
        if done:
            Q_est = r
        else:
            Q_est = r + self.gamma*self.Q(s_next).max()
        #print Q_est - self.Q(self.state, self.action)
        update[self.action] = Q_est - self.Q(self.state, self.action)
        td = update[self.action]
        self.policy.update(update, self.state, n=n)        
        self.state = s_next
        #print td, self.Q(self.state, self.action)
        return td

    def save(self, s_dir, **kwargs):
        self.policy.save(s_dir, **kwargs)
        self.Q = self.policy.Qfunction # make sure that the Q function is updated

    def load(self, l_dir, **kwargs):
        self.policy.load(l_dir, **kwargs)

class DoubleQLearning(Agent):
    def __init__(self, S, A, Qa=None, Qb=None, gamma=0.99, lr=1e-3, **kwargs):
        self.S = S
        self.A = A
        if Qa == None or Qb == None:
            if (Qa == None) + (Qb == None) == 1:
                print 'Warning you only supplied one of two required Qfunctions.'
            self.Qa = TableQ(S, A, lr=lr)
            self.Qb = TableQ(S, A, lr=lr)
        else:
            self.Qa = Qa
            self.Qb = Qb
        if policy == None:
            self.policy = ConsensusQ([self.Qa, self.Qb])
        self.Qa = self.policy.Qfunctions[0] # make sure that the Q function is updated
        self.Qb = self.policy.Qfunctions[1] # make sure that the Q function is updated

        self.gamma = gamma

        super(QLearning, self).__init__(self.policy, **kwargs)

    def observe(self, s_next, r, done, n):
        update = np.zeros(self.A.n)
        if done:
            Q_est = r
        else:
            if np.random.rand() > 0.5:
                idx = 0
                Q_est = r + self.gamma*self.Qb(s_next).max()
                update[self.action] = Q_est - self.Qa(self.state, self.action)
            else:
                idx = 1
                Q_est = r + self.gamma*self.Qa(s_next).max()
                update[self.action] = Q_est - self.Qb(self.state, self.action)
        td = update[self.action]
        self.policy.update(update, self.state, idx=idx, n=n)
        self.state = s_next
        return td

class DQN(Agent):
    def __init__(self, S, A, gamma=0.99, Qfunction=None, model=None, loss='mse', optimizer='adam', 
                 memory_size=10000, target_update_freq=1000, batch_size= 32, update_freq=4, 
                 action_repeat=4, history_len=4, image=False, **kwargs):
        self.S = S
        self.A = A

        if Qfunction == None:
            self.Q = KerasQ(S, A, model, loss=loss, optimizer=optimizer)
            if hasattr(self.Q, 'model'):
                self.targetQ = self.Q
                config = self.Q.model.get_config()
                model = Sequential.from_config(config)
                model.set_weights(self.Q.model.get_weights())
                self.targetQ.model = model
                 
            else:   
                self.targetQ = copy.deepcopy(self.Q)
            #self.targetQ = KerasQ(S, A, model, loss=loss, optimizer=optimizer)
        else:
            self.Q = Qfunction
            self.targetQ = copy.deepcopy(self.Q)

        super(DQN, self).__init__(**kwargs)
    
        if self.policy == None:
            self.policy = MaxQ(self.Q)
        assert hasattr(self.policy, 'Qfunction'), 'Policy must use Qfunctions.'
        self.Q = self.policy.Qfunction # make sure that the Q function is updated

        self.memory = deque([])

        self.gamma = gamma
        self.memory_size = memory_size
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.action_repeat = action_repeat
        self.history_len = history_len
        self.image = image

        self.frames = []
        self.frame_count = 0
        self.action_count = 0
        self.sumr = 0
        self.loss = 0
        self.h_count = 0

        self.action = self.A.sample()
        self.prev_state = self.S.sample()
        if self.image:
            if K.image_dim_ordering() == 'th': 
                self.state = np.concatenate([self.preprocess(self.S.sample()) for i in xrange(self.history_len)], axis=1)
            else:
                self.state = np.concatenate([self.preprocess(self.S.sample()) for i in xrange(self.history_len)], axis=-1)
        else:
            self.state = self.preprocess(self.S.sample())

    def observe(self, s_next, r, done, n):
        self.frame_count += 1

        # update the target network
        if n % self.target_update_freq == 0:
            self.targetQ.model.set_weights(self.Q.model.get_weights())
        
        # preprocess the state, handling images specially
        if self.image:
            if self.h_count < self.history_len != 0:
                self.frames.append(self.preprocess(s_next))
                self.prev_state = s_next 
                self.sumr += r
                self.h_count += 1
                return self.loss
            if K.image_dim_ordering() == 'th':    
                next_state = np.concatenate(self.frames, axis=1)
            else:
                next_state = np.concatenate(self.frames, axis=-1)
            r = self.sumr
            self.sumr = 0

            self.h_count = 0
            self.frames = []
        else:
            next_state = self.preprocess(s_next)

        # add transition to replay memory            
        transition = (self.state, self.action, next_state, r)
        if len(self.memory) < self.memory_size:
            self.memory.append(transition)
        else:
            self.memory.popleft()
            self.memory.append(transition)

        # update the policy
        if self.action_count % self.update_freq == 0 and self.frame_count > self.random_start:
            indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
            states = []
            actions = []
            s_nexts = []
            rs = []
            for i in xrange(indices.shape[0]):
                state, action, s_next, r = self.memory[indices[i]]
                states.append(state)
                actions.append(action)
                s_nexts.append(s_next)
                rs.append(r)
            states = np.concatenate(states, axis=0)
            actions = np.array(actions)
            s_nexts = np.concatenate(s_nexts, axis=0)
            rs = np.array(rs)

            targets = np.zeros((self.batch_size, self.A.n))
            targets = self.Q(states)
            if done:
                targets[np.arange(self.batch_size), actions] = rs
            else:
                Qs = self.targetQ(s_nexts)
                targets[np.arange(self.batch_size), actions] = rs + self.gamma*Qs.max(1)
                self.loss = self.policy.update(targets, states)

        self.state = next_state
        return self.loss

    def act(self, **kwargs):
        if self.frame_count % self.action_repeat == 0:
            # count how many actions you have selected
            self.action_count += 1
            # for less than random_start apply a random policy
            if self.frame_count < self.random_start:
                self.action = self.A.sample()
            else:
                self.action = self.policy(self.state, **kwargs)
        return self.action
    
    def preprocess(self, state):
        if self.image:
            # I need to process the state with the previous state
            # I take the max pixel value of the two frames
            state = np.maximum(state, self.prev_state)
            state = np.asarray(state, dtype=np.float32)
            # convert rgb image to yuv image
            yCbCr = Image.fromarray(state, 'YCbCr')
            # extract the y channel
            y, Cb, Cr = yCbCr.split()
            # rescale to 84x84
            y_res = y.resize((84,84))
            output = image.img_to_array(y_res)
        elif type(self.S) == gym.spaces.Discrete:
            # one-hot encoding
            output = np.zeros(self.S.n)
            output[state] = 1
        else:
            output = state

        return np.expand_dims(output, axis=0) #add axis for batch dimension

    def save(self, s_dir='.'):
        #self.Q.save('{}/Qmodel.h5'.format(s_dir))
        self.targetQ.save(s_dir, name='targetQmodel')
        self.policy.save(s_dir)
        pkl.dump(self.memory, open('{}/memory.pkl'.format(s_dir),'w'))

        session = {'frames':self.frames, 'frame_count':self.frame_count, 'action_count':self.action_count,
                   'sumr':self.sumr, 'loss':self.loss, 'h_count':self.h_count,
                   'action':self.action, 'prev_state':self.prev_state, 'state':self.state}
        pkl.dump(session, open('{}/session.pkl'.format(s_dir),'w'))

    def load(self,s_dir='.', **kwargs):
        self.targetQ.load(s_dir, name='targetQmodel', **kwargs)
        self.policy.load(s_dir, **kwargs)
        self.Q = self.policy.Qfunction # make sure that the Q function is updated

        self.memory = pkl.load(open('{}/memory.pkl'.format(s_dir), 'r'))
        session = pkl.load(open('{}/session.pkl'.format(s_dir),'r'))
        for key, value in session.iteritems():
            if hasattr(self, key):
                self.__dict__[key] = value
        

class CrossEntropy(Agent):
    def __init__(self, S, A, n_sample=100, top_p=0.2, init_mean=None, init_std=None, **kwargs):
        self.S = S
        self.A = A

        if policy == None:
            self.policy = Linear(A, S, degree=1)
        else:
            self.policy = policy
        self.param_dim = self.policy.param_dim

        self.n_sample = n_sample
        self.top_p = top_p
        self.n_elite = int(top_p * n_sample)

        if init_mean == None:
            self.mean = np.random.uniform(low=-1,high=1,size=self.param_dim)
            
        if init_std == None:
            self.std = np.random.uniform(low=0,high=1,size=self.param_dim) 

        super(CrossEntropy, self).__init__(self.policy, **kwargs)

        self.i = 0 #sample counter varies between 0 and n_sample-1
        self.R = np.zeros(self.n_sample) # the collection of rewards

    def observe(self, s_next, r, done, n):
        s_next = self.preprocess(s_next)
        # collect n parameter samples
        if self.i == 0:
            self.params = np.random.multivariate_normal(self.mean, np.diag(self.std), self.n_sample)

        # evaluate the parameters
        self.R[self.i] += r

        # select the elite set if all params have been evaluated
        if self.i >= self.n_sample-1:
            print 'Selecting Elite...'
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

    def preprocess(self, state):
        if type(self.S) == gym.spaces.Discrete:
            # one-hot encoding
            s = np.zeros(self.S.n)
            s[state] = 1
            return s
        else:
            return state
            
