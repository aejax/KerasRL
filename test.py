from agent import *
from value import *
from keras.optimizers import *
import keras.backend as K
import gym
import timeit
import argparse
import matplotlib.pyplot as plt

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
        env.monitor.start('tmp/{}'.format(agent.name), force=True)

    count_steps = 0
    try:
        for episode in xrange(n_episode):
            observation = env.reset()
            done = False
            l_sum = agent.observe(observation, 0, done, count_steps)
            r_sum = 0
            timer = 0    
            while not done:
                count_steps += 1

                if render and (episode+1)%log_freq == 0:
                    env.render(mode='human')

                action = agent.act(n=count_steps-10000)
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

def test_session(env_name, n_episode, interactive):
    env = gym.make(env_name)
    S = env.observation_space
    A = env.action_space

    if interactive:
        render = yntotf(raw_input('Render? [y/n] '))
        monitor = yntotf(raw_input('Monitor? [y/n] '))
        plot = yntotf(raw_input('Plot? [y/n] '))
    else:
        render = False
        monitor = False
        plot = True

    # define the save directory
    s_dir = env_name
    import os
    import os.path
    file_path = './' + s_dir
    if not os.path.exists(file_path):
        os.mkdir(s_dir)


    n_episode = n_episode
    tMax = 10000
    log_freq = 10

    def huber_loss(y_true, y_pred):
        err = y_pred - y_true
        return K.mean(K.sqrt(1 + K.square(err)) - 1, axis=-1)

    # Define your agent
    #agent = CrossEntropy(S,A)
    memory_size =       10000
    replay_start_size = 10000
    loss = 'mse'
    opt = 'adam' #RMSprop(lr=0.00025)
    name = 'DQN-atari'
    
    if atari:

    else:
        model = Sequential()
        if type(S) == gym.spaces.Discrete:
            model.add(Dense(100, input_dim=S.n, activation='relu', name='l1'))
        else:
            model.add(Dense(100, input_shape=S.shape, activation='relu', name='l1'))
        model.add(Dense(20, activation='relu', name='l2'))
        model.add(Dense(A.n, activation='linear', name='l3'))

    Q = KerasQ(S, A, model=model, loss=loss, optimizer=opt)
    policy = MaxQ(Q, randomness=SAepsilon_greedy(A, epsilon=0.1, final=1000))
    agent = DQN(S, A, policy=policy, model=model, memory_size=memory_size, replay_start_size=replay_start_size, batch_size=32,
                target_update_freq=1000, update_freq=1, action_repeat=1, history_len=1, name=name)

    #knn = KNNQ(S, A, n_neighbors=5, memory_size=1000000, memory_fit=100)
    #agent = QLearning(S, A, Q=knn, name='KNN')

    # Perform test
    begin = timeit.default_timer()
    run(env, agent, n_episode, tMax, log_freq, render, monitor, plot, s_dir)
    end = timeit.default_timer()
    dt = end - begin
    print 'Run time: {:}min {:.3}s'.format(dt // 60, dt % 60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL testing script.')
    parser.add_argument('-e', '--environment', type=str, default='FrozenLake-v0')
    parser.add_argument('-n', '--n_episode', type=int, default=100)
    parser.add_argument('-i', '--interactive', action='store_true')

    args = parser.parse_args()
    print args.n_episode
    test_session(args.environment, args.n_episode, args.interactive)
