import numpy as np
import matplotlib.pyplot as plt
import timeit

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.optimizers import sgd

import gym

class SimpleAgent(object):
    def __init__(self, S, A, gamma=0.9, epsilon=0.1, learning_rate=1e-3):
        self.S = S
        self.A = A
        #self.hidden = hidden
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate

        self.input_dim = self.get_space_dim(S)
        self.output_dim = self.get_space_dim(A)

        model = Sequential()
        model.add(Dense(self.output_dim, input_dim=self.input_dim, name='Layer1'))

        model.compile(loss='mse', optimizer=sgd(lr=learning_rate))
    
        self.model = model
        self.state = np.zeros((1, self.input_dim))
        self.action = 0
    
    def observe(self, s_next, r, done):
        s_next = s_next[np.newaxis,:] # turn vector into matrix

        targets = np.zeros((1,self.output_dim))
        inputs = self.state

        targets = self.model.predict(self.state)

        if done:
            targets[0,self.action] = r
        else:
            maxQ = self.model.predict(s_next).max()
            targets[0,self.action] = r + self.gamma*maxQ
        loss = self.model.train_on_batch(inputs, targets)

        self.state = s_next

        return loss

    def act(self):
        rand = np.random.uniform()
        if rand > self.epsilon:
            self.action = self.model.predict(self.state).argmax()
        else:
            self.action = np.random.choice(self.output_dim)
        return self.action            

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

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model = load_model(filename)

def run(env, agent, n_episode, tMax, plot=True):
    returns = []
    losses = []
    start_time = timeit.default_timer()
    for episode in xrange(n_episode):
        observation = env.reset()
        l_sum = agent.observe(observation, 0, False )
        r_sum = 0
            
        for t in xrange(tMax):
            #env.render(mode='human')
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
    
    if plot:
        plt.figure()
        plt.subplot(121)
        plt.plot(returns)
        plt.title('Rewards')
        plt.subplot(122)
        plt.plot(losses)
        plt.title('Loss')
        plt.savefig('lr={:.2}.png'.format(agent.learning_rate), format='png')
        plt.close('all')

    return ave_r

def test():
    n_episode = 100000
    tMax = 200

    gamma = 0.9
    epsilon = 0.1
    learning_rate = 1e-2

    env = gym.make('CartPole-v0')
    S = env.observation_space
    A = env.action_space
    
    agent = SimpleAgent(S, A, gamma=gamma, epsilon=epsilon, learning_rate=learning_rate)
    #agent.model = load_model('model.h5')

    ave_r = run(env, agent, n_episode, tMax, plot=True)

    agent.model.save('model.h5')
    exit()

    # Sample learning rates #
    lrs = 10**np.random.uniform(-2.0, -1.0, size=10)
    learning_rates = [lrs[n] for n in xrange(lrs.shape[0]) ]

    ave_returns = []
    for lr in learning_rates:
        agent = SimpleAgent(S, A, gamma=gamma, epsilon=epsilon, learning_rate=lr)
        ave_r = run(env, agent, n_episode, tMax, plot=True)
        ave_returns.append(ave_r)

    plt.figure()
    plt.semilogx(learning_rates, ave_returns, 'o')
    plt.savefig('ave_returns.png'.format(), format='png')
    

if __name__ == '__main__':
    test()
