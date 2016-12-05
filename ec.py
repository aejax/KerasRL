import theano
import theano.tensor as T
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, merge

from agent import MaxQ, EpisodicControl, SAepsilon_greedy
from value import TableQ2

def get_agent(env, name=None):
    S = env.observation_space
    A = env.action_space

    random_start = 1000
    epsilon = 0.01
    exploration_frames = 2000

    gamma = 0.99
    memory_size = 5000
    mode = 'action_tables'
    embedding_function = None
    embedding_dim = 4
    image = False
    name = 'EC' if name == None else name

    Q = TableQ2(S, A, maxlen=memory_size, mode=mode, embedding_dim=embedding_dim)
    policy = MaxQ(Q, randomness=SAepsilon_greedy(A, epsilon=epsilon, final=exploration_frames))
    #policy = MaxQ(Q)
    agent = EpisodicControl(S, A, policy=policy, embedding_function=embedding_function, gamma=gamma, image=image, random_start=random_start, name=name)

    return agent

def load(l_dir, env, name=None):
    agent = get_agent(env, name=name)

    agent.load(l_dir)
    return agent

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    agent = get_agent(env)
    print agent.__dict__
