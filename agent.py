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

    def update(self, update, state, **kwargs):
        return self.Qfunction.update(update, state, **kwargs)

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
        td = Q_est - self.Q(self.state, self.action)
        if self.Q.update_mode == 'set':
            update[self.action] = Q_est
        else:
            update[self.action] = td
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
    def __init__(self, S, A, gamma=0.99, Qfunction=None, targetQfunction=None, model=None, loss='mse', optimizer='adam', 
                 memory_size=10000, target_update_freq=1000, batch_size= 32, update_freq=4, 
                 history_len=4, image=False, double=True, bounds=False, update_cycles=None, **kwargs):

        super(DQN, self).__init__(**kwargs)
        self.S = S
        self.A = A

        if self.policy == None:
            if Qfunction == None:
                assert model != None
                Qfunction = KerasQ(S, A, model, loss=loss, optimizer=optimizer, bounds=bounds)
            if targetQfunction == None:
                assert model != None
                if Qfunction == None:
                    config = model.get_config()
                    if type(model) == Model:
                        model2 = Model.from_config(config)
                    elif type(model) == Sequential:
                        model2 = Sequential.from_config(config)
                    model2.set_weights(model.get_weights())
                else:
                    model2 = model
                targetQfunction = KerasQ(S, A, model2, loss=loss, optimizer=optimizer, bounds=bounds)
            self.policy = MaxQ(Qfunction)
            self.Q = self.policy.Qfunction # make sure that the Q function is updated
            self.targetQ = targetQfunction
    
        else:
            assert hasattr(self.policy, 'Qfunction'), 'Policy must use Qfunctions.'
            if targetQfunction == None:
                assert model != None
                targetQfunction = KerasQ(S, A, model, loss=loss, optimizer=optimizer, bounds=bounds)
            self.Q = self.policy.Qfunction # make sure that the Q function is updated
            self.targetQ = targetQfunction

        self.memory = deque(maxlen=memory_size) #deque handles length

        self.gamma = gamma
        self.memory_size = memory_size
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.history_len = history_len
        self.image = image
        self.double = double
        self.bounds = bounds
        self.update_cycles = update_cycles

        self.frame_count = 0
        self.loss = 0
        self.t = 0

        self.Qforward_time = 0
        self.policy_update_time = 0
        self.observation_time = 0
        self.act_time = 0
        self.get_batch_time = 0
        self.get_bounds_time = 0
        self.get_future_time = 0
        self.get_past_time = 0

        self.action = self.A.sample()
        self.prev_state = self.S.sample()
        self.r = 0
        self.done = False
        if self.bounds:
            self.Umin = np.zeros((1,A.n))
            self.Lmax = np.zeros((1,A.n))
            

        if self.image:
            if K.image_dim_ordering() == 'th': 
                self.state = np.concatenate([self._preprocess(self.S.sample()) for i in xrange(self.history_len)], axis=1)
            else:
                self.state = np.concatenate([self._preprocess(self.S.sample()) for i in xrange(self.history_len)], axis=-1)
        else:
            self.state = self._preprocess(self.S.sample())
        if self.bounds:
            self.Q([self.state,self.Umin,self.Lmax])
            self.targetQ([self.state,self.Umin,self.Lmax])
        else:
            self.Q(self.state)
            self.targetQ(self.state)
        self.state = self.prev_state

    def observe(self, s_next, r, done, n):
        start0 = timeit.default_timer()
        self.frame_count += 1
        self.t += 1

        # update the target network
        if n % self.target_update_freq == 0:
            self.targetQ.model.set_weights(self.Q.model.get_weights())

        next_state = self._preprocess(s_next)
        self.prev_state = s_next

        # update the policy
        start = timeit.default_timer()
        if self.frame_count % self.update_freq == 0 and self.frame_count > self.random_start: 
            if self.bounds:
                states, actions, s_nexts, rs, dones, Umin, Lmax = self._get_batch()

                targets = np.zeros((self.batch_size, self.A.n))
                start = timeit.default_timer()
                targets = self.Q([states,Umin,Lmax])
                end = timeit.default_timer()
                self.Qforward_time += end - start

                if self.double:
                    _a = self.Q([s_nexts,Umin,Lmax]).argmax(1)
                    update = rs + self.gamma*self.targetQ([s_nexts,Umin,Lmax])[np.arange(self.batch_size),_a]
                else:
                    Qs = self.targetQ([s_nexts,Umin,Lmax])
                    update = rs + self.gamma*Qs.max(1)
                targets[np.arange(self.batch_size), actions] = np.where(dones, rs, update)
                start = timeit.default_timer()
                self.loss = self.policy.update(targets, [states,Umin,Lmax], n=self.update_cycles)
                end = timeit.default_timer()
                self.policy_update_time += end - start   
            else:      
                states, actions, s_nexts, rs, dones = self._get_batch()
 
                targets = np.zeros((self.batch_size, self.A.n))
                start = timeit.default_timer()
                targets = self.Q(states)
                end = timeit.default_timer()
                self.Qforward_time += end - start

                if self.double:
                    _a = self.Q(s_nexts).argmax(1)
                    update = rs + self.gamma*self.targetQ(s_nexts)[np.arange(self.batch_size),_a]
                else:
                    start = timeit.default_timer() 
                    Qs = self.targetQ(s_nexts)
                    update = rs + self.gamma*Qs.max(1)
                targets[np.arange(self.batch_size), actions] = np.where(dones, rs, update)
                start = timeit.default_timer()
                self.loss = self.policy.update(targets, states, n=self.update_cycles) 
                end = timeit.default_timer()
                self.policy_update_time += end - start        

        if self.bounds and done:
            nextR = 0
            for idx in range(self.t-1):
                R = self.memory[-idx-1][2] + self.gamma*nextR
                self.memory[-idx-1].append(R)
                nextR = R
            self.t = 0

        self.state = next_state
        self.r = r
        self.done = done
        end = timeit.default_timer()
        self.observation_time += end - start0
        return self.loss

    def act(self, **kwargs):
        start = timeit.default_timer()
        # for less than random_start apply a random policy
        if self.frame_count < self.random_start:
            self.action = self.A.sample()
        else:
            idx = len(self.memory)-self.history_len
            state = self._get_history(idx)
            if self.bounds:
                self.Umin = np.zeros((1,self.A.n))
                self.Lmax = np.zeros((1,self.A.n))
                self.action = self.policy([state,self.Umin,self.Lmax], **kwargs)
            else:
                self.action = self.policy(state, **kwargs)

        transition = [self.state, self.action, self.r, self.done]
        self.memory.append(transition)
        end = timeit.default_timer()
        self.act_time += end - start
        return self.action

    def _get_memory_size(self):
        transition = self.memory[-1]
        size = transition[0].nbytes
        size *= len(self.memory)
        return size
    
    def _preprocess(self, state):
        if self.image:
            # I need to process the state with the previous state
            # I take the max pixel value of the two frames
            #state[0] = np.maximum(state[0], self.prev_state[0])
            #state[1] = np.maximum(state[1], self.prev_state[1])
            #state[2] = np.maximum(state[2], self.prev_state[2])
            # convert rgb image to yuv image
            yCbCr = Image.fromarray(state, 'YCbCr')
            # extract the y channel
            y, Cb, Cr = yCbCr.split()
            # rescale to 84x84
            y_res = y.resize((84,84))
            output = image.img_to_array(y_res)
            output = np.array(output, dtype=np.uint8) #keep memory small
        elif type(self.S) == gym.spaces.Discrete:
            # one-hot encoding
            output = np.zeros(self.S.n)
            output[state] = 1
        else:
            output = state

        return np.expand_dims(output, axis=0) #add axis for batch dimension

    def _get_batch(self):
        # assume that the memory is organized as follows:
        #   first a state is recorded,
        #   then an action based on that state is recorded,
        #   then a reward is recorded,
        #   followed by a subsequent state.
        # e.g. (r0, s0, a0, r1, s1, a1, r2, ..., sN-1, aN-1, rN)
        # the current state input is of length history_len,
        # history_len = 4: state = (s0, s1, s2, s3)
        # the current action is that associated with the last four states,
        # state = (s0, s1, s2, s3); action = (a3); reward = (r4)
        # what about the next state? The next state is what the agent sees next:
        # next_state = (s1, s2, s3, s4)
        # what about end of episode cases? i.e. state = (sN-4, sN-3, sN-2, sN-1)
        # here the next state that the agent gets is from the next episode (s0, s1, s2, s3)
        # in this case we ignore the next state and just use the reward in the update
        # we need to include a end of episode flag, (s, a, r, d), in the memory
        start = timeit.default_timer()
        if self.bounds:
            arange = np.arange(4, len(self.memory) - self.history_len - 2 - self.t)
        else:
            arange = np.arange(len(self.memory) - self.history_len -1)
        indices = np.random.choice(arange, self.batch_size, replace=False)
        states = []
        actions = []
        s_nexts = []
        rs = []
        dones = []
        if self.bounds:
            Us = []
            Ls = []
        for i in xrange(indices.shape[0]):
            idx = indices[i]
            # if done get a new index
            m_slice = [self.memory[i] for i in xrange(idx, idx + self.history_len)]
            done = reduce(lambda x,y: x or y, [memory[3] for memory in m_slice])
            while done:
                idx = np.random.randint(len(self.memory) - self.history_len -1)
                m_slice = [self.memory[i] for i in xrange(idx, idx + self.history_len)]
                done = reduce(lambda x,y: x or y, [memory[3] for memory in m_slice])
            
            state = self._get_history(idx)
            action = self.memory[idx + self.history_len][1]
            next_state = self._get_history(idx+1)
            reward = self.memory[idx + self.history_len + 1][2]
            done =   self.memory[idx + self.history_len + 1][3]

            states.append(state)
            actions.append(action)
            s_nexts.append(next_state)
            rs.append(reward)
            dones.append(done)
            if self.bounds:
                umin, lmax = self._get_bounds(idx, state, action, k=4)
                Us.append(umin)
                Ls.append(lmax)
     
        states = np.concatenate(states, axis=0)
        actions = np.array(actions)
        s_nexts = np.concatenate(s_nexts, axis=0)
        rs = np.array(rs)
        rs = np.clip(rs, -1, 1)
        dones = np.array(dones)
        if self.bounds:
            Umin = np.array(Us)
            Lmax = np.array(Ls)
            # timer
            end = timeit.default_timer()
            self.get_batch_time += end - start
            return states, actions, s_nexts, rs, dones, Umin, Lmax
        else:
            # timer
            end = timeit.default_timer()
            self.get_batch_time += end - start
            return states, actions, s_nexts, rs, dones

    def _get_history(self, idx):
        m_slice = [self.memory[i] for i in xrange(idx, idx + self.history_len)]
        state = [memory[0] for memory in m_slice]
        if K.image_dim_ordering() == 'th':    
            state = np.concatenate(state, axis=1)
        else:
            state = np.concatenate(state, axis=-1)
        return state

    def _get_future(self, idx, k=4):
        start = timeit.default_timer()
        m_slice = [self.memory[i] for i in xrange(idx, idx + k + 1)]
        R = np.array([t[4] for t in m_slice])
        s = np.concatenate([self._get_history(i) for i in xrange(idx + 1, idx + k + 1)], axis=0)
        end = timeit.default_timer()
        self.get_future_time += end - start
        return R, s

    def _get_past(self, idx, k=4):
        start = timeit.default_timer()
        m_slice = [self.memory[i] for i in xrange(idx - k, idx + 1)]
        R = np.array([t[4] for t in m_slice])
        a = np.array([t[1] for t in m_slice])[:-1]
        s = np.concatenate([self._get_history(i) for i in xrange(idx - k, idx)], axis=0)
        end = timeit.default_timer()
        self.get_past_time += end - start
        return R, s, a

    def _get_bounds(self, idx, state, action, k=4):
        start = timeit.default_timer()
        self.Umin = np.zeros((k,self.A.n))
        self.Lmax = np.zeros((k,self.A.n))
        Q = self.Q([state, np.zeros((1,self.A.n)), np.zeros((1,self.A.n))])
        Umin = np.array(Q[0]) #to vector
        Lmax = np.array(Q[0]) #to vector
        R, s = self._get_future(idx, k)
        L = R[0] + self.gamma**np.arange(1,k+1) * (self.targetQ([s, self.Umin, self.Lmax]).max(-1) - R[1:])
        R, s, a = self._get_past(idx, k)
        U = self.gamma**(-np.arange(1,k+1)[::-1]) * (self.targetQ([s, self.Umin, self.Lmax])[np.arange(k),a] - R[:-1]) + R[-1]
        Umin[action] = U.min()
        Lmax[action] = L.max()
        # We need U and L to be equal to y_pred except for at action a:
        #   e.g. U = [y_pred[0], y_pred[1], 23, y_pred[3]]
        end = timeit.default_timer()
        self.get_bounds_time += end - start
        return Umin, Lmax

    def times(self):
        n = self.frame_count - self.random_start
        m = (self.frame_count - self.random_start) // self.update_freq
        if n <= 0:
            self.act_time = self.observation_time = 0
        print 'Act Time: \t\t\t', self.act_time / n
        print 'Observation Time: \t\t', self.observation_time / n
        print '\tGet Batch Time: \t', self.get_batch_time / m
        print '\tGet Bounds Time: \t', self.get_bounds_time / m
        print '\tGet Future Time: \t', self.get_future_time / m
        print '\tGet Past Time: \t\t', self.get_past_time / m
        print '\tQ Forward Pass Time: \t', self.Qforward_time / m
        print '\tQ Backward Pass Time: \t', self.policy_update_time / m


    def save(self, s_dir):
        import os
        import os.path
        s_dir = s_dir + '/' + self.name
        file_path = './' + s_dir
        if not os.path.exists(file_path):
            os.mkdir(s_dir)
        self.targetQ.save(s_dir, name='targetQmodel')
        self.policy.save(s_dir)
        pkl.dump(self.memory, open('{}/memory.pkl'.format(s_dir),'w'))

        session = {'frame_count':self.frame_count, 'loss':self.loss, 'action':self.action, 
                   'prev_state':self.prev_state, 'state':self.state}
        pkl.dump(session, open('{}/session.pkl'.format(s_dir),'w'))

    def load(self, s_dir, **kwargs):
        s_dir = s_dir + '/' + self.name
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
        super(CrossEntropy, self).__init__(**kwargs)

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

"""
class EpisodicControl(Agent):
    def __init__(self, S, A, embedding_function=None, Qfunction=None):
        super(EpisodicControl, self).__init__(**kwargs)

        self.S = S
        self.A = A

        if embedding_function == None:
            s_dim = get_space_dim(self.S)
            if s_dim > 10:
                RM = np.random.random((10, s_dim))
                self.phi = lambda o: np.dot(RM, o.flatten())
            else:
                self.phi = lambda o: o
        else:
            self.phi = embedding_function
   
        if self.policy == None:
            if Qfunction == None:
                Qfunction = KNNQ(S, A)
            self.policy == MaxQ(Qfunction)
        assert hasattr(self.policy, 'Qfunction'), 'Policy must use Qfunctions.'
        self.Q = self.policy.Qfunction

        self.rewards = []
        self.states = []
        self.actions = []

        self.state  = self.phi(self._preprocess(self.S.sample()))
        self.action = self.A.sample()

    def observe(self, s_next, r, done):
        if not done:
            self.state = self.phi(self._preprocess(s_next))
            self.states.append(self.state)
            self.rewards.append(r)
            return 0
        else:
            R_tp1 = 0
            for r, s, a in zip(reversed(self.rewards), reversed(self.states), reversed(self.actions)):
                R = r + self.gamma*R_tp1
                Q_ec(s, a) = R
                target = np.zeros(self.A.n)
                target[a] = R

    def _preprocess(self, state):
        if self.image:
            # I need to process the state with the previous state
            # I take the max pixel value of the two frames
            state[0] = np.maximum(state[0], self.prev_state[0])
            state[1] = np.maximum(state[1], self.prev_state[1])
            state[2] = np.maximum(state[2], self.prev_state[2])
            # convert rgb image to yuv image
            yCbCr = Image.fromarray(state, 'YCbCr')
            # extract the y channel
            y, Cb, Cr = yCbCr.split()
            # rescale to 84x84
            y_res = y.resize((84,84))
            output = image.img_to_array(y_res)
            output = np.array(output, dtype=np.uint8) #keep memory small
        elif type(self.S) == gym.spaces.Discrete:
            # one-hot encoding
            output = np.zeros(self.S.n)
            output[state] = 1
        else:
            output = state

        return np.expand_dims(output, axis=0) #add axis for batch dimension
"""
