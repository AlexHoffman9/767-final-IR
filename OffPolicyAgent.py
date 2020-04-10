import numpy as np
import gym
import tensorflow as tf
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import random_walk_env as RW
import time


#TODO: what is the target policy pi? need it to get importance ratio
    # for epsilon greedy it is easy to compute by taking max of values and giving f
    # their code mentions something about target policy. searching "target" and "Get_policy" in github yield results
# 


class OffPolicyAgent():
    # construct agent's model separately, so it can be sized according to problem
    def __init__(self, n_actions, n_replay, input_dim, out_dim, env, lr, discount):
        self.replay_buffer = [0]*n_replay # change to data structure that stores s,a,r,s'
        # self.model = self.build_model()
        self.actions = range(n_actions)
        self.lr = lr
        self.discount = discount
        self.model = self.build_model(input_dim, out_dim)
        print(self.model.summary())
        self.env = env

    # build neural network for state value function
    # I think a linear model would be plenty for the random walk example... so this could be changed
    def build_model(self, input_dim, out_dim):
        input_layer = Input(shape=(input_dim), name='state_input')
        returns = Input(shape=(1), name='returns_input')
        hidden_layer = Dense(32, activation = "relu", name='hidden_layer')(input_layer)
        output_layer = Dense(out_dim, activation="softmax", name='output_layer')(hidden_layer)
        # loss function for batch update
        opt = Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, amsgrad=True)
        model = Model(inputs=[input_layer], outputs=[output_layer])
        model.compile(loss="mean_squared_error", optimizer = opt)
        return model

    # complete episode of experience and then train using buffer
    # not sure how much experience to get before training on it...one episode? 2? n timesteps?
    def train(self):
        buffer_index = 0
        # init state
        s = self.env.reset()
        done = False
        while not done:
            # choose action according to policy
            a = np.random,choice(a=self.actions, p=self.policy(s))
            (s2,r,done,_) = env.step(a)
            # compute the desired state value = r + v(s2)
            #TODO: figure out what goes in the buffer, put it there
                # likely faster to get all experience, then do batch of prediction to get v(s'), then put the data in the official replay buffer
                # doing step->predict->add-to-buffer would be slow because of keras
            # self.model.predict([s], )
            # add example to buffer
            self.replay_buffer[buffer_index,:]
        # add 
        
    # need to choose tiling (will depend on child class for each environment
    # default tiling is for 10 state random walk 
    def construct_feature(self,s): 
        return [i == s for i in range(self.len(self.actions))]

    # uniform policy for now
    # could replace this with specific agent
    def policy(self, s):
        n=len(self.actions)
        return [1.0/n]*n


then = time.time()
env = RW.RandomWalkEnv(10)
lr=.001
discount=.99
agent = OffPolicyAgent(2,1000,2,1,env, lr, discount)
