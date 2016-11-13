import keras.backend as K
from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Flatten
from keras.optimizers import RMSprop

from agent import MaxQ, DQN, SAepsilon_greedy
from value import KerasQ

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

    memory_size = 1000000
    random_start = 50000 // frameskip
    exploration_frames = 1000000 // frameskip
    epsilon = 0.1
    gamma = 0.99            
    target_update_freq = 10000
    update_freq = 4
    history_len = 4
    batch_size = 32
    loss = huber_loss
    opt =  RMSprop(lr=0.00025)
    bounds = False
    double = False
    update_cycles = None
    name = 'DQN' if name == None else name
    image = True
 
    # define the model
    if K.image_dim_ordering() == 'th':
        state_shape = (4,84,84)
    else:
        state_shape = (84,84,4)
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, subsample=(4,4), input_shape=state_shape, activation='relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1,1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(A.n))

    config = model.get_config()
    model2 = Sequential.from_config(config)
    model2.set_weights(model.get_weights())

    Q = KerasQ(S, A, model=model, loss=loss, optimizer=opt, bounds=bounds, batch_size=batch_size)
    targetQ = KerasQ(S, A, gamma=gamma, model=model2, loss=loss, optimizer=opt, bounds=bounds, batch_size=batch_size)
    policy = MaxQ(Q, randomness=SAepsilon_greedy(A, epsilon=epsilon, final=exploration_frames))
    agent = DQN(S, A, policy=policy, Qfunction=Q, targetQfunction=targetQ, memory_size=memory_size, random_start=random_start, 
                batch_size=batch_size, target_update_freq=target_update_freq, update_freq=update_freq, history_len=history_len, 
                image=image, name=name, double=double, bounds=bounds, update_cycles=update_cycles)

    return agent

if __name__ == '__main__':
    import gym
    env = gym.make('Pong-v3')
    agent = get_agent(env)
    print agent.__dict__

