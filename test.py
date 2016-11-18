from agent import *
from value import *
from keras.optimizers import *
import keras.backend as K
import gym
import timeit
import argparse

import matplotlib
#Force matplotlib to use any Xwindows backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import simple_dqn
import atari_dqn
import matching_q

def yntotf(s):
    if s.lower() == 'y':
        return True
    elif s.lower() == 'n':
        return False
    else:
        print '\'{}\' cannot be converted to True or False.'.format(s)
        return None

def run(env, agent, n_episode, tMax, log_freq, render, monitor, plot, s_dir):
    returns = []
    losses = []
    if monitor:
        env.monitor.start('{}/{}'.format(s_dir, agent.name), force=True)

    if hasattr(agent, 'frame_count'):
        count_steps = agent.frame_count
    else:
        count_steps = 0
    try:
        for episode in xrange(n_episode):
            observation = env.reset()
            done = False
            l_sum = agent.observe(observation, 0, done, count_steps)
            count_steps += 1
            r_sum = 0
            timer = 0
            s = timeit.default_timer() 
            while not done:
                count_steps += 1

                if render and (episode+1)%log_freq == 0:
                    env.render(mode='human')

                action = agent.act(n=count_steps-agent.random_start)

                observation, reward, done, info = env.step(action)

                # End the episode at tMax
                timer += 1
                if timer == tMax:
                    done = True
                l_sum += agent.observe(observation, reward, done, count_steps)

                r_sum += reward
                if timer == tMax:
                    done = True

            if (episode+1)%log_freq == 0:
                print 'Episode {} finished with return of {}.'.format(episode+1,r_sum)
                e = timeit.default_timer()
                print 'Training step time: ', (e - s) / timer
                print 'Training steps: ', timer
                if hasattr(agent, 'times'):
                    agent.times()
            returns.append(r_sum)
            losses.append(l_sum / timer)

    except KeyboardInterrupt:
        pass

    if monitor:
        env.monitor.close()

    def movingaverage(values, window):
        weights = np.repeat(1.0, window)/window
        sma = np.convolve(values, weights, 'valid')
        return sma

    window_size = 100
    rMVA = movingaverage(returns,window_size)
    lMVA = movingaverage(losses,window_size)

    ave_r = reduce(lambda x, y: x+y, returns) / n_episode
    print 'Average Reward: {}'.format(ave_r)
    print 'Max 100 Episode Average Reward: {}'.format(rMVA.max())
    print 'Number of environment steps: {}'.format(count_steps)

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

def test_session(env_name, agent_name, n_episode, log_freq, interactive, l_dir, s_dir):
    env = gym.make(env_name)
    S = env.observation_space
    A = env.action_space

    #set defaults
    if s_dir == '':
        s_dir = env_name
    if l_dir == '':
        l_dir = s_dir

    render = False
    monitor = False
    plot = False
    save = False
    load = False

    if interactive:
        render = yntotf(raw_input('Render? [y/n] '))
        monitor = yntotf(raw_input('Monitor? [y/n] '))
        plot = yntotf(raw_input('Plot? [y/n] '))
        save = yntotf(raw_input('Save? [y/n] '))
        if save:
            tmp = raw_input('Save directory? (default is {}): '.format(s_dir))
            s_dir = tmp if tmp != '' else s_dir
            print s_dir
        load = yntotf(raw_input('Load? [y/n] '))
        if load:
            tmp = raw_input('Load directory? (default is {}): '.format(l_dir))
            l_dir = tmp if tmp != '' else l_dir
            print l_dir

    # define the save and load directory
    import os
    import os.path
    file_path = './' + s_dir
    if not os.path.exists(file_path):
        os.mkdir(s_dir)
    file_path = './' + l_dir
    if not os.path.exists(file_path):
        os.mkdir(l_dir)

    #define run length
    n_episode = n_episode
    tMax = env.spec.timestep_limit
    log_freq = log_freq

    #define agent
    if agent_name == 'simple_dqn':
        if load:
            agent = simple_dqn.load(l_dir, env, name='sDDQNb')
        else:
            agent = simple_dqn.get_agent(env, name='sDDQNb')
    elif agent_name == 'atari_dqn':
        if load:
            agent = atari_dqn.load(l_dir, env, name='DQN')
        else:
            agent = atari_dqn.get_agent(env, name='DQN')
    elif agent_name == 'matching_q':
        if load:
            agent = matching_q.load(l_dir, env, name='MQL')
        else:
            agent = matching_q.get_agent(env, name='MQL')
    else:
        raise ValueError, '{} is not a valid agent name.'.format(agent_name)

    #knn = KNNQ(S, A, n_neighbors=5, memory_size=100000, memory_fit=100, lr=1.0, weights='distance')
    #agent = QLearning(S, A, Q=knn, name='KNN-1', random_start=random_start)

    # Perform test
    print 'Beginning training for {} episodes.'.format(n_episode)
    begin = timeit.default_timer()
    run(env, agent, n_episode, tMax, log_freq, render, monitor, plot, s_dir)
    end = timeit.default_timer()
    dt = end - begin
    print 'Run time: {:}min {:.3}s'.format(dt // 60, dt % 60)

    if save:
        print 'Saving agent to {}'.format(s_dir)
        begin = timeit.default_timer()
        agent.save(s_dir)

        end = timeit.default_timer()
        dt = end - begin
        print 'Save time: {:}min {:.3}s'.format(dt // 60, dt % 60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL testing script.')
    parser.add_argument('-e', '--environment', type=str, default='FrozenLake-v0')
    parser.add_argument('-a', '--agent', type=str, default='simple_dqn')
    parser.add_argument('-n', '--n_episode', type=int, default=100)
    parser.add_argument('--log', type=int, default=10)
    parser.add_argument('-l', '--load_dir', type=str, default='')
    parser.add_argument('-s', '--save_dir', type=str, default='')
    parser.add_argument('-i', '--interactive', action='store_true')

    args = parser.parse_args()
    test_session(args.environment, args.agent, args.n_episode, args.log, args.interactive, args.load_dir, args.save_dir)
