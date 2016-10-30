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

def yntotf(s):
    if s.lower() == 'y':
        return True
    elif s.lower() == 'n':
        return False
    else:
        print '\'{}\' cannot be converted to True or False.'.format(s)
        return None

def run(env, agent, n_episode, tMax, log_freq, render, monitor, plot, s_dir, random_start):
    returns = []
    losses = []
    if monitor:
        env.monitor.start('tmp/{}'.format(agent.name), force=True)

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
            while not done:
                count_steps += 1

                if render and (episode+1)%log_freq == 0:
                    env.render(mode='human')

                action = agent.act(n=count_steps-random_start)
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

def test_session(env_name, n_episode, interactive, l_dir):
    env = gym.make(env_name)
    S = env.observation_space
    A = env.action_space

    if interactive:
        render = yntotf(raw_input('Render? [y/n] '))
        monitor = yntotf(raw_input('Monitor? [y/n] '))
        plot = yntotf(raw_input('Plot? [y/n] '))
        save = yntotf(raw_input('Save? [y/n] '))
        load = yntotf(raw_input('Load? [y/n] '))
    else:
        render = False
        monitor = False
        plot = True
        save = False
        load = False

    # define the save and load directory
    s_dir = env_name
    import os
    import os.path
    file_path = './' + s_dir
    if not os.path.exists(file_path):
        os.mkdir(s_dir)
    if l_dir != '':
        load = True
        l_dir = ldir
    elif l_dir == '' and load:
        l_dir = s_dir

    n_episode = n_episode
    tMax = 200
    log_freq = 10

    def huber_loss(y_true, y_pred):
        err = y_pred - y_true
        return K.mean(K.sqrt(1 + K.square(err)) - 1, axis=-1)

    # Define your agent
    #agent = CrossEntropy(S,A)
    memory_size =  100000
    random_start = 50000
    exploration_frames = 1000000
    loss = 'mse'
    opt = 'adam' #RMSprop(lr=0.00025)
    name = 'DQN-atari'
    atari = False       
        
    if atari:
        if K.image_dim_ordering() == 'th':
            state_shape = (4,84,84)
        else:
            state_shape = (84,84,4)
        # default DQN for atari
        model = Sequential()
        model.add(Convolution2D(32, 8, 8, subsample=(4,4), input_shape=state_shape, activation='relu'))
        model.add(Convolution2D(64, 4, 4, subsample=(2,2), activation='relu'))
        model.add(Convolution2D(64, 3, 3, subsample=(1,1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(A.n))
    else:
        model = Sequential()
        if type(S) == gym.spaces.Discrete:
            model.add(Dense(100, input_dim=S.n, activation='relu', name='l1'))
        else:
            model.add(Dense(100, input_shape=S.shape, activation='relu', name='l1'))
        model.add(Dense(20, activation='relu', name='l2'))
        model.add(Dense(A.n, activation='linear', name='l3'))

    #Q = KerasQ(S, A, model=model, loss=loss, optimizer=opt)
    #policy = MaxQ(Q, randomness=SAepsilon_greedy(A, epsilon=0.1, final=exploration_frames))
    #agent = DQN(S, A, policy=policy, model=model, memory_size=memory_size, random_start=random_start, batch_size=32,
    #            target_update_freq=10000, update_freq=4, action_repeat=4, history_len=4, image=True, name=name)

    knn = KNNQ(S, A, n_neighbors=5, memory_size=100000, memory_fit=100, lr=1.0)
    agent = QLearning(S, A, Q=knn, name='KNN-1', random_start=random_start)

    if load:
        print 'Loading agent from {}'.format(l_dir)
        begin = timeit.default_timer()
        agent.load(l_dir, loss=loss, optimizer=opt)

        end = timeit.default_timer()
        dt = end - begin
        print 'Load time: {:}min {:.3}s'.format(dt // 60, dt % 60)

    # Perform test
    print 'Beginning training for {} episodes.'.format(n_episode)
    begin = timeit.default_timer()
    run(env, agent, n_episode, tMax, log_freq, render, monitor, plot, s_dir, random_start)
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
    parser.add_argument('-n', '--n_episode', type=int, default=100)
    parser.add_argument('-l', '--load_dir', type=str, default='')
    parser.add_argument('-i', '--interactive', action='store_true')

    args = parser.parse_args()
    test_session(args.environment, args.n_episode, args.interactive, args.load_dir)
