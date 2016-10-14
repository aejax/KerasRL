import numpy as np
import matplotlib.pyplot as plt
import gym
from gym.envs.registration import EnvSpec
import timeit

from keras.layers import Dense, Activation
from keras.models import Sequential

from agents import QLearning, DoubleQLearning, KerasDQN, CrossEntropy

def run(env, agent, n_episode, tMax, plot=True, log_freq=10, render=False, monitor=False, **kwargs):

    if 'epsilon' in kwargs:
        min_epsilon = kwargs['epsilon']
    else:
        min_epsilon = 0.0

    if 'anneal_percent' in kwargs:
        percent = kwargs['anneal_percent']
    else:
        percent = 0.0

    def anneal(episode):
        epsilon = (n_episode - (1.0/percent)*episode) / n_episode
        return max(epsilon, min_epsilon)

    returns = []
    losses = []
    start_time = timeit.default_timer()

    if monitor:
        env.monitor.start('tmp/{}'.format(agent.name), force=True)

    try:

        for episode in xrange(n_episode):
            observation = env.reset()
            done = False
            l_sum = agent.observe(observation, 0, done )
            r_sum = 0
            timer = 0    
            while not done:
                if render and (episode+1)%log_freq == 0:
                    env.render(mode='human')
                if percent == 0.0:
                    action = agent.act(epsilon=min_epsilon)
                else:
                    action = agent.act(epsilon=anneal(episode))
                observation, reward, done, info = env.step(action)

                # End the episode at tMax
                timer += 1
                if timer == tMax:
                    done = True

                l_sum += agent.observe(observation, reward, done)
                r_sum += reward
                if timer == tMax:
                    done = True

            if (episode+1)%log_freq == 0:
                print 'Episode {} finished with return of {}.'.format(episode+1,r_sum)
            returns.append(r_sum)
            losses.append(l_sum)

    except KeyboardInterrupt:
            pass

    if monitor:
        env.monitor.close()

    end_time = timeit.default_timer()
    ave_r = reduce(lambda x, y: x+y, returns) / n_episode
    #print 'Learning Rate: {:.2}'.format(agent.learning_rate)
    print "Training time: {}".format(end_time - start_time)
    print 'Average Reward: {}'.format(ave_r)

    #print agent.Q(None)

    #print 'Q_A'
    #print agent.Q_A(None)
    #print 'Q_B'
    #print agent.Q_B(None)

    def movingaverage(values, window):
        weights = np.repeat(1.0, window)/window
        sma = np.convolve(values, weights, 'valid')
        return sma

    window_size = 100
    rMVA = movingaverage(returns,window_size)
    lMVA = movingaverage(losses,window_size)

    print 'Max 100 Episode Average Reward: {}'.format(rMVA.max())   

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
        plt.savefig('{}.png'.format(agent.name), format='png')
        plt.close('all')

    return ave_r

def test_session():
    #AgentClass = DoubleQLearning
    #AgentClass = QLearning
    #AgentClass = KerasDQN
    AgentClass = CrossEntropy

    #model = Sequential()
    #model.add(Dense(output_dim=10, input_dim=4))
    #model.add(Activation('sigmoid'))
    #model.add(Dense(output_dim=2))

    #model.compile(loss='mse',
    #              optimizer='adam')

    single_run = True
    plot = True
    render = False
    monitor = True

    #task_name = 'CartPole-v0'
    #task_name = 'MountainCar-v0'
    #task_name = 'FrozenLake-v0'
    task_name = 'LunarLander-v2'

    env = gym.make(task_name)

    S = env.observation_space
    A = env.action_space

    n_episode = 100000
    tMax = 1000
    log_freq = 1000

    learning_rate=1e-4
    gamma=0.99

    if single_run:
        #agent = AgentClass(S, A, learning_rate=learning_rate, gamma=gamma)
        #agent = AgentClass(S, A, model, gamma=gamma)
        agent = AgentClass(S, A, n_sample=1000, top_p=0.02)

        ave_r = run(env, agent, n_episode, tMax, plot=plot, epsilon=0.1,
                    log_freq=log_freq, render=render, monitor=monitor, anneal_percent=1.0)
        
    else:
        # Sample learning rates #
        lrs = 10**np.random.uniform(-4.0, -2, size=10)
        learning_rates = [lrs[n] for n in xrange(lrs.shape[0]) ]

        ave_returns = []
        for lr in learning_rates:
            agent = AgentClass(S, A, learning_rate=lr, gamma=gamma)

            ave_r = run(env, agent, n_episode, tMax, plot=plot)
            print agent.table
            ave_returns.append(ave_r)

        plt.figure()
        plt.semilogx(learning_rates, ave_returns, 'o')
        plt.savefig('lr_returns.png'.format(), format='png')
        plt.xlabel('learning rate')
        plt.ylabel('average returns')

if __name__ == '__main__':
    test_session()
