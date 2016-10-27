import numpy as np
import matplotlib.pyplot as plt
import gym
from gym.envs.registration import EnvSpec
import timeit
import argparse

from keras.layers import Dense, Activation
from keras.models import Sequential

from agents import QLearning, DoubleQLearning, KerasDQN, CrossEntropy

def run(env, agent, n_episode, tMax, plot=True, log_freq=10, render=False, monitor=False, s_dir='.', **kwargs):

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
        plt.savefig('{}/{}.png'.format(s_dir, agent.name), format='png')
        plt.close('all')

    return ave_r

class Session(object):
    def __init__(self, agent_class, task_name, single_run=True, plot=True, render=False,
                 monitor=False, save_agent=False, load_agent=False, n_episode=100, tMax=100,
                 epsilon=0.1, anneal_percent=0.0, log_freq=10, s_dir=None):
        self.agent_class = agent_class
        self.task_name = task_name

        self.env = gym.make(task_name)

        self.S = self.env.observation_space
        self.A = self.env.action_space

        self.single_run = single_run
        self.plot = plot
        self.render = render
        self.monitor = monitor
        self.save_agent = save_agent
        self.load_agent = load_agent

        self.n_episode = n_episode
        self.tMax = tMax
        self.epsilon = epsilon
        self.log_freq = log_freq
        self.anneal_percent = anneal_percent

        self.s_dir = s_dir
        if self.s_dir == None:
            self.s_dir = task_name
            import os
            import os.path
            file_path = './' + self.s_dir
            if not os.path.exists(file_path):
                os.mkdir(self.s_dir)

        self.configure_agent()

    def run(self):
        ave_r = run(self.env, self.agent, self.n_episode, self.tMax, plot=self.plot, epsilon=self.epsilon,
                    log_freq=self.log_freq, render=self.render, monitor=self.monitor, 
                    anneal_percent=self.anneal_percent, s_dir=self.s_dir)
        return ave_r

    def set_conditions(self):
        self.single_run = yntotf(raw_input('Single run? [y/n] '))
        self.plot = yntotf(raw_input('Plot? [y/n] '))
        self.render = yntotf(raw_input('Render? [y/n] '))
        self.monitor = yntotf(raw_input('Monitor? [y/n] '))
        self.save_agent = yntotf(raw_input('Save the agent? [y/n] '))
        self.load_agent = yntotf(raw_input('Load the agent? [y/n] '))

        v = raw_input('Set # of episodes [default is {}]: '.format(self.n_episode))
        if v != '':
            self.n_episode = eval(v) 

        v = raw_input('Set tMax [default is {}]: '.format(self.tMax))
        if v != '':
            self.tMax = eval(v)        

        v = raw_input('Set log frequency [default is {}]: '.format(self.log_freq))
        if v != '':
            self.log_freq = eval(v)

        v = raw_input('Set epsilon [default is {}]: '.format(self.epsilon))
        if v != '':
            self.epsilon = eval(v)

        v = raw_input('Set anneal percent [default is {}]: '.format(self.anneal_percent))
        if v != '':
            self.anneal_percent = eval(v)

        v = raw_input('Set save directory [default is {}]: '.format(self.s_dir))
        if v != '':
            self.s_dir = v

    def configure_agent(self):
        # Configure the agent
        agent_config = self.agent_class.get_config()
        for key, value in agent_config.iteritems():
            v = raw_input('Value for {} [default is {}]: '.format(key, value))
            if type(value) != str:
                if v != '':
                    agent_config[key] = eval(v)
            else:
                if v != '':
                    agent_config[key] = v

        self.agent = self.agent_class(self.S, self.A, **agent_config)

def yntotf(s):
    if s.lower() == 'y':
        return True
    elif s.lower() == 'n':
        return False
    else:
        print '\'{}\' cannot be converted to True or False.'.format(s)
        return None
        

def test_session(agent, environment, interactive):

    if agent == 'doubleQ':
        AgentClass = DoubleQLearning
    elif agent == 'Q':
        AgentClass = QLearning
    elif agent == 'DQN':
        AgentClass = KerasDQN
    elif agent == 'crossentropy':
        AgentClass = CrossEntropy
    else:
        print '{} is not a valid agent name.'.format(agent)

    #model = Sequential()
    #model.add(Dense(output_dim=10, input_dim=4))
    #model.add(Activation('sigmoid'))
    #model.add(Dense(output_dim=2))

    #model.compile(loss='mse',
    #              optimizer='adam')

    #task_name = 'CartPole-v0'
    #task_name = 'MountainCar-v0'
    #task_name = 'FrozenLake-v0'
    #task_name = 'LunarLander-v2'
    #task_name = 'Pong-v0'
    #task_name = 'Roulette-v0'
    task_name = environment

    """
    # Define the test conditions
    if interactive:
        print 'Entering interactive mode.'
        sess = Session(AgentClass, task_name)
        sess.set_conditions()
    else:
        single_run = True
        plot = True
        render = False
        monitor = False
        save_agent = False
        load_agent = False

        # define the save directory
        s_dir = task_name

        import os
        import os.path
        file_path = './' + s_dir
        if not os.path.exists(file_path):
            os.mkdir(s_dir)

        env = gym.make(task_name)

        S = env.observation_space
        A = env.action_space
        
        n_episode = 10000
        tMax = 100
        log_freq = 100

        #lr_mode = 'constant'
        #learning_rate = 1e-2
        lr_mode = 'variable'    
        learning_rate = lambda n: 1.0 / pow(n,0.8)
        gamma=0.99
    """

    sess = Session(AgentClass, task_name)

    if interactive:
        sess.set_conditions()

    sess.run()

    exit()

    if single_run:           
        agent = AgentClass(S, A, learning_rate=learning_rate, gamma=gamma, lr_mode=lr_mode)
        #agent = AgentClass(S, A, model=None, gamma=gamma)
        #agent = AgentClass(S, A, n_sample=1000, top_p=0.02)

        if load_agent:
            agent.load(s_dir+'/model.h5', s_dir+'/target_model.h5', s_dir+'/states.npy',
                       s_dir+'/next_states.npy', s_dir+'/actions.npy', s_dir+'/rewards.npy', s_dir+'/config.pkl')

        ave_r = run(env, agent, n_episode, tMax, plot=plot, epsilon=0.1,
                    log_freq=log_freq, render=render, monitor=monitor, anneal_percent=0.1, s_dir=s_dir)

        if save_agent:
            print agent.frame_count
            agent.save(dir=s_dir)

        print agent.Q_B(0).max()

        #print 'Policy'
        #a_names = ['  Up ', 'Right', ' Down', ' Left']
        #for i in xrange(S.n):
        #    if (i+1) % 4 == 0:
        #        print a_names[agent.Q_A(i).argmax()]
        #    else:
        #        print a_names[agent.Q_A(i).argmax()], 
        #    #print i, agent.Q(i), a_names[agent.Q(i).argmax()]

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
    parser = argparse.ArgumentParser(description='RL testing script.')
    parser.add_argument('-a', '--agent', type=str, default='crossentropy')
    parser.add_argument('-e', '--environment', type=str, default='FrozenLake-v0')
    parser.add_argument('-i', '--interactive', action='store_true')

    args = parser.parse_args()
    test_session(args.agent, args.environment, args.interactive)
