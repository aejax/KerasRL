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

class QTable(object):
    def __init__(self, S, A, learning_rate=1e-3, gamma=0.9, policy=epsilon_greedy):
        self.S = S
        self.A = A
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.policy = policy

        assert type(S) == gym.spaces.Discrete
        assert type(A) == gym.spaces.Discrete

        self.space_dim = self.get_space_dim(S)
        self.action_dim = self.get_space_dim(A)

        self.table = np.zeros((self.space_dim, self.action_dim))
        
        self.state = 0
        self.action = 0

    def observe(self, s_next, r, done):
        if done:
            Q_est = r
        else:
            Q_est = r + self.gamma*self.table[s_next].max()
        update = Q_est - self.table[self.state, self.action]
        self.table[self.state,self.action] += self.learning_rate*update

        self.state = s_next

        return update

    def act(self, **kwargs):
        action = self.table[self.state].argmax()
        self.action = self.policy(action, self.A, **kwargs)
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
            print 'Wrong type for space: must be either Box or Discrete'
            
        return s_dim  

def run(env, agent, n_episode, tMax, plot=True, **kwargs):
    returns = []
    losses = []
    start_time = timeit.default_timer()
    for episode in xrange(n_episode):
        observation = env.reset()
        l_sum = agent.observe(observation, 0, False )
        r_sum = 0
            
        for t in xrange(tMax):
            #env.render(mode='human')
            action = agent.act(**kwargs)
            observation, reward, done, info = env.step(action)

            l_sum += agent.observe(observation, reward, done)
            r_sum += reward
            if done:
                break

        if (episode+1)%100 == 0:
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
        plt.savefig('qtable_data/lr{:.2}.png'.format(agent.learning_rate), format='png')
        plt.close('all')

    return ave_r

def flQ_table(table):
    #scale the table
    table = table / table.sum(1)[:,np.newaxis]

    n_state = table.shape[0]
    n_row = int(np.sqrt(n_state).round())
    n_col = n_row

    flQ_table = np.zeros((3*n_row, 3*n_col))
    for state in xrange(n_state):
        tile = np.zeros((3,3))
        tile[0,1] = table[state,0]
        tile[1,0] = table[state,3]
        tile[1,1] = 1 #state
        tile[1,2] = table[state,1]
        tile[2,1] = table[state,2]

        row = state//n_row
        col = state%n_row
        flQ_table[3*row:3*row+3, 3*col:3*col+3] = tile
    return flQ_table

def test_qtable():

    single_run = True

    n_episode = 1000000
    tMax = 200

    env = gym.make('FrozenLake-v0')

    S = env.observation_space
    A = env.action_space

    learning_rate=1e-2
    gamma=0.99
    policy=epsilon_greedy

    if single_run:
        agent = QTable(S, A, learning_rate=learning_rate, gamma=gamma, policy=policy)
        ave_r = run(env, agent, n_episode, tMax, plot=True, epsilon=0.1)

        plt.matshow(flQ_table(agent.table))
        plt.savefig('qtable_data/policy.png', format='png')
        
    else:
        # Sample learning rates #
        lrs = 10**np.random.uniform(-4.0, -2, size=10)
        learning_rates = [lrs[n] for n in xrange(lrs.shape[0]) ]

        ave_returns = []
        for lr in learning_rates:
            agent = QTable(S, A, learning_rate=lr, gamma=gamma, policy=policy)

            ave_r = run(env, agent, n_episode, tMax, plot=True)
            print agent.table
            ave_returns.append(ave_r)

        plt.figure()
        plt.semilogx(learning_rates, ave_returns, 'o')
        plt.savefig('lr_returns.png'.format(), format='png')
        plt.xlabel('learning rate')
        plt.ylabel('average returns')

if __name__ == '__main__':
    test_qtable()
