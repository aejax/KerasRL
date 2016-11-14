import keras.backend as K
from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Flatten
from keras.optimizers import RMSprop

from agent import MaxQ, DQN, SAepsilon_greedy
from value import KerasQ

import gym

def huber_loss(y_true, y_pred):
    err = y_pred - y_true
    return K.mean(K.sqrt(1 + K.square(err)) - 1, axis=-1)

def get_agent(env, name=None):
    S = env.observation_space
    A = env.action_space

    if hasattr(env, 'frameskip'):
        frameskip = env.frameskip
        if isinstance(frameskip, tuple):
            valid_frames = range(frameskip[0], frameskip[1])
            frameskip = reduce(lambda x, y: x+y, valid_frames) // len(valid_frames)
    else:
        frameskip = 1

    memory_size = 50000
    random_start = 500 // frameskip
    exploration_frames = 10000 // frameskip
    epsilon = 0.01
    gamma = 0.99               
    target_update_freq = 1000
    update_freq = 1
    history_len = 1
    batch_size = 32
    loss = huber_loss
    opt =  RMSprop(lr=0.00025)
    bounds = True
    double = True
    update_cycles = None
    name = 'DQN' if name == None else name
    image = False
 
    # define the model
    model = Sequential()
    if type(S) == gym.spaces.Discrete:
        model.add(Dense(50, input_dim=S.n, activation='relu', name='l1'))
    else:
        model.add(Dense(50, input_shape=S.shape, activation='relu', name='l1'))
    model.add(Dense(100, activation='relu', name='l2'))
    model.add(Dense(50, activation='relu', name='l3'))
    model.add(Dense(A.n, activation='linear', name='l4'))

    config = model.get_config()
    model2 = Sequential.from_config(config)
    model2.set_weights(model.get_weights())

    Q = KerasQ(S, A, model=model, loss=loss, optimizer=opt, bounds=bounds, batch_size=batch_size)
    targetQ = KerasQ(S, A, model=model2, loss=loss, optimizer=opt, bounds=bounds, batch_size=batch_size)
    policy = MaxQ(Q, randomness=SAepsilon_greedy(A, epsilon=epsilon, final=exploration_frames))
    agent = DQN(S, A, gamma=gamma, policy=policy, Qfunction=Q, targetQfunction=targetQ, memory_size=memory_size, random_start=random_start, 
                batch_size=batch_size, target_update_freq=target_update_freq, update_freq=update_freq, history_len=history_len, 
                image=image, name=name, double=double, bounds=bounds, update_cycles=update_cycles)

    return agent

def load(l_dir, env, name=None):
    agent = get_agent(env, name=name)

    loss = huber_loss
    opt =  RMSprop(lr=0.00025)
    agent.load(l_dir, loss=loss, optimizer=opt, custom_objects={'bounded_loss':agent.Q.loss})
    return agent

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    agent = get_agent(env)
    print agent.__dict__
