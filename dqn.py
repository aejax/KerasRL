from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.optimizers import sgd, adam

import numpy as np
import matplotlib.pyplot as plt
import gym
import timeit

def epsilon_greedy(action, A, epsilon=0.1):
    assert type(A) == gym.spaces.Discrete
    rand = np.random.random()
    if rand > epsilon:
        return action
    else:
        return np.random.choice(A.n)

class ReplayMemory(object):
    def __init__(self, memory_size, state_size):
        self.memory_size = memory_size

        self.states = np.zeros((memory_size, state_size))
        self.actions = np.zeros((memory_size,), dtype='int32')
        self.next_states = np.zeros((memory_size, state_size))
        self.rewards = np.zeros((memory_size,))

        self.idx = 0 #memory location
        self.full = False

    def add(self, transition):
        state, action, next_state, reward = transition        

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
    

class DQN(object):
    def __init__(self, S, A, learning_rate=1e-3, gamma=0.9, policy=epsilon_greedy, batch_size=32, update_freq=1000, memory_size=100):
        self.S = S
        self.A = A
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.policy = policy
        self.update_freq = update_freq
        self.batch_size = batch_size

        self.input_dim = self.get_space_dim(S)
        self.output_dim = self.get_space_dim(A)

        ################################
        # Build Model and Target Model #
        ################################
        model = Sequential()
        model.add(Dense(10, input_dim=self.input_dim, name='layer1'))
        model.add(Activation('relu'))
        model.add(Dense(self.output_dim, name='layer2'))

        model.compile(loss='mse', optimizer=sgd(lr=self.learning_rate))
        self.model = model
    
        self.target_model = Sequential.from_config(self.model.get_config())
        self.target_model.set_weights(self.model.get_weights())

        # Build Replay Memory #
        self.replay_memory = ReplayMemory(memory_size, self.input_dim)

        self.state = np.zeros(self.input_dim)
        self.action = 0

        self.time = 0
        self.episode = 0

    def observe(self, s_next, r, done):
        s_next = s_next[np.newaxis,:] # turn vector into matrix

        self.done = done
        self.time_management(done)

        transition = (self.state, self.action, s_next, r)
        self.replay_memory.add(transition)
        minibatch = self.replay_memory.get_minibatch(self.batch_size)
        
        loss = self.batch_updates(minibatch)

        self.state = s_next

        return loss

    def act(self, **kwargs):
        action = self.model.predict(self.state).argmax()
        self.action = self.policy(action, self.A, **kwargs)
        return self.action

    def batch_updates(self, minibatch):
        targets = np.zeros((min(self.batch_size,self.time), self.output_dim))
        states, actions, next_states, rewards = minibatch

        targets = self.model.predict(states)

        if self.done:
            targets[actions] = rewards[:,np.newaxis]
        else:
            Qs = self.target_model.predict(next_states)
            targets[actions] = (rewards + self.gamma * Qs.argmax(1))[:,np.newaxis]
        loss = self.model.train_on_batch(states, targets)
        return loss

    def time_management(self, done):
        self.time += 1
        if done:
            self.episode += 1 
        if self.time % self.update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())
        

    @staticmethod
    def get_space_dim(Space):
        # get the dimensions of the spaces
        if type(Space) == gym.spaces.Box:
            s_dim = 1
            for n in Space.shape:
                s_dim *= n
        elif type(Space) == gym.spaces.Discrete:
            s_dim = Space.n
        else:
            print 'Wrong type for A: must be either Box or Discrete'
            
        return s_dim

def run(env, agent, n_episode, tMax, plot=True):
    returns = []
    losses = []
    start_time = timeit.default_timer()
    for episode in xrange(n_episode):
        observation = env.reset()
        l_sum = agent.observe(observation, 0, False )
        r_sum = 0
            
        for t in xrange(tMax):
            env.render(mode='human')
            action = agent.act()
            observation, reward, done, info = env.step(action)

            l_sum += agent.observe(observation, reward, done)
            r_sum += reward
            if done:
                break

        if (episode+1)%10 == 0:
            print 'Episode {} finished with return of {}.'.format(episode+1,r_sum)
        returns.append(r_sum)
        losses.append(l_sum)

    end_time = timeit.default_timer()
    ave_r = reduce(lambda x, y: x+y, returns) / n_episode
    print 'Learning Rate: {:.2}'.format(agent.learning_rate)
    print "Training time: {}".format(end_time - start_time)
    print 'Average Reward: {}'.format(ave_r)

    def movingaverage(values, window):
        weights = np.repeat(1.0, window)/window
        sma = np.convolve(values, weights, 'valid')
        return sma

    window_size = 100
    rMVA = movingaverage(returns,window_size)
    lMVA = movingaverage(losses,window_size)
    
    if plot:
        plt.figure()
        plt.subplot(121)
        plt.plot(rMVA)
        plt.title('Rewards')
        plt.xlabel('Average rewards per {} episodes'.format(window_size))
        plt.subplot(122)
        plt.plot(lMVA)
        plt.title('Loss')
        plt.xlabel('Average losses per {} episodes'.format(window_size))
        plt.savefig('dqn_lr{:.2}.png'.format(agent.learning_rate), format='png')
        plt.close('all')

    return ave_r

def test_dqn():
    import os
    filename = 'dqn.h5'
    single_run = True
    load = False
    save = False

    n_episode = 100
    tMax = 200

    env = gym.make('CartPole-v0')

    S = env.observation_space
    A = env.action_space

    learning_rate=5e-3
    gamma=0.99
    policy=epsilon_greedy
    batch_size=32
    update_freq=1000
    memory_size=10000

    agent = DQN(S, A, learning_rate=learning_rate, gamma=gamma, policy=policy,
                    batch_size=batch_size, update_freq=update_freq, memory_size=memory_size)
    initial_weights = agent.model.get_weights()

    if single_run:
        agent = DQN(S, A, learning_rate=learning_rate, gamma=gamma, policy=policy,
                    batch_size=batch_size, update_freq=update_freq, memory_size=memory_size)

        if load:
            if os.path.isfile(filename):
                agent.model = load_model(filename)
                agent.target_model = load_model(filename)

        ave_r = run(env, agent, n_episode, tMax, plot=False)
        
        if save:
            agent.model.save(filename)

    else:
        # Sample learning rates #
        lrs = 10**np.random.uniform(-4.0, -2, size=10)
        learning_rates = [lrs[n] for n in xrange(lrs.shape[0]) ]

        ave_returns = []
        for lr in learning_rates:
            agent = DQN(S, A, learning_rate=lr, gamma=gamma, policy=policy,
                    batch_size=batch_size, update_freq=update_freq, memory_size=memory_size)
            agent.model.set_weights(initial_weights)
            agent.target_model.set_weights(initial_weights)

            ave_r = run(env, agent, n_episode, tMax, plot=True)
            ave_returns.append(ave_r)

        plt.figure()
        plt.semilogx(learning_rates, ave_returns, 'o')
        plt.savefig('lr_returns.png'.format(), format='png')
        plt.xlabel('learning rate')
        plt.ylabel('average returns')
    

if __name__ == '__main__':
    test_dqn()
