import theano
import theano.tensor as T
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, merge, Flatten

from agent import MaxQ, MatchingQLearning, SAepsilon_greedy
from value import KerasQ

#theano.config.optimizer = 'fast_compile'

def batch_sim(w, M, eps=1e-6):
    """
    w: matrix with shape (batch, memory_elem)
    M: tensor with shape (batch, memory_size, memory_elem)
    eps: numerical stability parameter
    """
    M = M.dimshuffle(1,0,2) # (N, batch, M)
    def cos_sim(u, v, eps=eps):
        """
        Takes two vectors and calculates the scalar cosine similarity.

        u: vector with shape (memory_elem,)
        v: vector with shape (memory_elem,)
        returns: scalar
        """
        sim = T.dot(u,v) / T.sqrt((u*u).sum() * (v*v).sum() + eps)
        return sim

    def batch_cos_sim(m_i, w):
        """
        Takes two matrices and calculates the scalar cosine similarity
        of their columns.

        m_i: matrix with shape (batch, memory_elem)
        w: matrix with shape (batch, memory_elem)
        returns: vector with shape (batch,)
        """
        sim, _ = theano.map(fn=cos_sim, sequences=[w, m_i])
        return sim

    sim, _ = theano.map(fn=batch_cos_sim, sequences=[M], non_sequences=[w])
    sim = sim.dimshuffle(1,0) # (batch, memory_size)
    return sim

def batch_sim2(w, M, eps=1e-6):
    """
    w: matrix with shape (batch, memory_elem)
    M: matrix with shape (memory_size, memory_elem)
    eps: numerical stability parameter
    """
    M = M.dimshuffle(1,0) # (memory_elem, memory_size)
    batch_sim = T.dot(w, M) #(batch, memory_size)
    return batch_sim

def batch_sim3(w, M, eps=1e-6):
    """
    w: matrix with shape (batch, memory_elem)
    M: tensor with shape (batch, memory_size, memory_elem)
    eps: numerical stability parameter
    """
    M = M[0] #only one true memory
    M = M.dimshuffle(1,0) # (memory_elem, memory_size)
    batch_sim = T.dot(w, M) #(batch, memory_size)
    return batch_sim

def batch_sim4(w, M, eps=1e-6):
    """
    w: matrix with shape (batch, memory_elem)
    M: tensor with shape (batch, memory_size, memory_elem)
    eps: numerical stability parameter
    """
    M = M[0] #only one true memory
    #M = M.dimshuffle(1,0) # (memory_elem, memory_size)
    def norm(A):
        """
        Calculate the column norm of matrix A
        A: matrix with shape (N, M)
        return: vector with shape (N,)
        """
        return T.sqrt(T.dot(A,A.T).diagonal())

    norm = T.outer(norm(w), norm(M)) #(batch, memory_size)
    batch_sim = T.dot(w, M.T) / (norm + eps) #(batch, memory_size)
    return batch_sim

def batch_sim5(w, M, eps=1e-6):
    """
    w: matrix with shape (batch, memory_elem)
    M: tensor with shape (batch, memory_size, memory_elem)
    eps: numerical stability parameter
    """
    M = M[0] # (memory_size, memory_elem)
    def batch_cos_sim(m, w, eps=eps):
        """
        Takes two vectors and calculates the scalar cosine similarity.

        m: vector with shape (memory_elem,)
        w: vector with shape (batch, memory_elem)
        returns: scalar
        """
        sim = T.dot(m,w.T) / T.sqrt((m*m).sum() * (w*w).sum(1) + eps)
        return sim #(batch,)

    sim, _ = theano.map(fn=batch_cos_sim, sequences=[M], non_sequences=[w])
    sim = sim.dimshuffle(1,0) # (batch, memory_size)
    return sim

def batch_sim6(w, M, eps=1e-6):
    """
    w: matrix with shape (batch, memory_elem)
    M: tensor with shape (batch, memory_size, memory_elem)
    eps: numerical stability parameter
    """
    M = M[0] #only one true memory
    #M = M.dimshuffle(1,0) # (memory_elem, memory_size)
    def norm(A):
        """
        Calculate the column norm of matrix A
        A: matrix with shape (N, M)
        return: vector with shape (N,)
        """
        n, _ = theano.map(fn=lambda a: T.sqrt((a*a).sum()), sequences=[A])
        return n

    norm = T.outer(norm(w), norm(M)) #(batch, memory_size)
    batch_sim = T.dot(w, M.T) / (norm + eps) #(batch, memory_size)
    return batch_sim

def attention(a_dim):
    def Att(tensors):
        emb, memory = tensors
        a = T.nnet.softmax(batch_sim6(emb, memory[:,:,:-a_dim]))
        return T.dot(a, memory[0,:,-a_dim:])
    return Att

def ignore(y_true, y_pred):
    return T.zeros((1,))

def get_agent(env, name=None):
    S = env.observation_space
    A = env.action_space
    print S.shape
    random_start = 2000
    epsilon = 0.01
    exploration_frames = 50000

    gamma = 0.99
    memory_size = 5000
    embedding_size = 100
    memory_shape = (memory_size, embedding_size + A.n)
    history_len = 1
    image = False
    batch_size = 128
    train = True
    loss = ['mse', ignore]
    opt = 'adam'
    name = 'MQL' if name == None else name

    #Define the model      
    state = Input(shape=S.shape)
    memory = Input(shape=memory_shape)
    ind = Input(shape=(1,memory_size))
    #h1 = Dense(10, activation='relu')(state)
    #emb = Dense(embedding_size, activation='linear')(h1)
    flat = Flatten()(state)
    emb = Dense(embedding_size, activation='linear')(flat)

    out = merge([emb, memory], mode=attention(A.n), output_shape=(A.n,))
    model = Model(input=[state, memory], output=[out,emb])

    Q = KerasQ(model=model, loss=loss, optimizer=opt)
    policy = MaxQ(Q, randomness=SAepsilon_greedy(A, epsilon=epsilon, final=exploration_frames))
    agent = MatchingQLearning(S, A, policy=policy, gamma=gamma, memory_size=memory_size,
                              embedding_size=embedding_size, history_len=history_len, image=image, 
                              batch_size=batch_size, train=train, random_start=random_start, name=name)

    return agent

def load(l_dir, env, name=None):
    agent = get_agent(env, name=name)

    loss = 'mse'
    opt =  'adam'
    agent.load(l_dir, loss=loss, optimizer=opt)
    return agent

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    agent = get_agent(env)
    print agent.__dict__
