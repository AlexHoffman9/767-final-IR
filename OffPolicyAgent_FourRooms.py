import numpy as np
import gym
import tensorflow as tf
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from random_walk_env import RandomWalkEnv
from four_rooms_env import FourRoomsEnv
import time
from OffPolicyAgent import OffPolicyAgent
tf.compat.v1.disable_eager_execution()


# get(dict,key) returns value for key


# Agent class with defaults for importance sampling
# can write new classes for WIS, IR, etc. by overwriting key functions
# Compatible with problem_name='RandomWalk' or 'FourRooms'
class OffPolicyAgent_FourRooms(OffPolicyAgent):
    # construct agent's model separately, so it can be sized according to problem
    def __init__(self, problem_name, n_replay, env, target_policy, behavior_policy, lr, discount, IS_method='IS'):
        self.lr = lr
        self.discount = discount
        self.n_replay = n_replay
        self.env = env
        self.actions = range(4)
        self.n_features = 11
        self.model = self.build_model(self.n_features*2, 1, IS_method)
        print(self.model.summary())
        self.target_policy = target_policy # 2d array indexed by state, action
        self.behavior_policy = behavior_policy
        self.replay_buffer = np.zeros(shape=(n_replay), dtype=[('s',(np.int32,2)), ('s2',(np.int32,2)), ('r',np.int32), ('ratio', np.float)]) # state, next state, reward, ratio
        self.t=0
        self.name = IS_method

    # build neural network for state value function
    # Default is single layer linear layer
    def build_model(self, input_dim, out_dim, IS_method):
        input_layer = Input(shape=(input_dim), name='state_input')
        ratios = Input(shape=(1), name='importance_ratios')
        hidden_layer = Dense(32, activation = "relu", name='hidden_layer')(input_layer)
        output_layer = Dense(out_dim, activation="linear", name='output_layer')(hidden_layer)
        # output_layer = Dense(out_dim, activation="linear", name='output_layer')(input_layer) #(hidden_layer)
        # loss function for batch update
        # just MSE loss multiplied by importance sampling ratio
        def is_loss(y_true, y_pred):
            se = tf.math.multiply(tf.math.square(y_true-y_pred), ratios) # weights loss according to sampling ratio. If ratio=0, sample is essentially ignored
            return tf.math.reduce_mean(se)
        def wis_minibatch_loss(y_true,y_pred):
            ratio_sum = tf.reduce_sum(ratios)
            se = tf.math.multiply(tf.math.square(y_true-y_pred), ratios) # weights loss according to sampling ratio. If ratio=0, sample is essentially ignored
            return tf.math.reduce_sum(se)/ratio_sum
        def wis_buffer_loss(y_true,y_pred):
            buffer_entries = np.min(self.t,self.n_replay)
            ratio_sum = np.sum(self.replay_buffer['ratio'][0:buffer_entries]) # only sum entries in buffer up to current size of buffer
            k = len(ratios)
            se = tf.math.multiply(tf.math.square(y_true-y_pred), ratios) # weights loss according to sampling ratio. If ratio=0, sample is essentially ignored
            return buffer_entries*tf.math.reduce_sum(se)/(ratio_sum*k) # n/k * (sum errors / sum ratios) 
        # opt = Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, amsgrad=True)
        model = Model(inputs=[input_layer, ratios], outputs=[output_layer])
        if IS_method == 'IS':
            model.compile(loss=is_loss, optimizer = SGD(lr=self.lr))
        elif IS_method == 'WIS_minibatch':
            model.compile(loss=wis_minibatch_loss, optimizer = SGD(lr=self.lr))
        elif IS_method == 'WIS_buffer':
            model.compile(loss=wis_buffer_loss, optimizer = SGD(lr=self.lr))
        return model

    # instead of generating one episode of experience, take 16 steps of experience
    def generate_episode(self, k=16):
        # init state
        s = self.env.reset()
        done = False
        steps = 0 # counting to k steps
        while steps < k:
            # choose action according to policy
            a = np.random.choice(a=self.actions, p=self.behavior_policy[s[0],s[1]])
            (s2,r,done,_) = self.env.step(a)
            ratio = self.target_policy[s[0],s[1],a] / self.behavior_policy[s[0],s[1],a]
            self.replay_buffer[self.t%self.n_replay] = (s,s2,r,ratio)
            s=s2
            self.t += 1
            steps += 1
            if done:
                s = self.env.reset()
                done = False

    # do batch of training using replay buffer
    # Default is to do a minibatch update. The paper uses both minibatch and incremental updates, so this could be changed
    def train_batch(self, n_samples, batch_size):
        sample_indices = self.sample_buffer(n_samples)
        # compute targets = ratio*(r + v(s'))
        rewards = self.replay_buffer['r'][sample_indices]
        next_state_features = self.construct_features(self.replay_buffer['s2'][sample_indices])
        state_features = self.construct_features(self.replay_buffer['s'][sample_indices])
        ratios = self.replay_buffer['ratio'][sample_indices]
        next_values = self.model.predict([next_state_features, np.zeros(next_state_features.shape[0])]).flatten()
        # v(s') is zero for terminal state, so need to fix model prediction
        for i in range(n_samples):
            # if experience ends in terminal state then s==s2
            if (self.replay_buffer['s'][sample_indices[i]] ==  self.replay_buffer['s2'][sample_indices[i]]).all():
                next_values[i] = 0.0
        targets = (rewards + self.discount*next_values)
        self.model.fit([state_features, ratios], targets, batch_size=batch_size, verbose=0)

    # need to choose tiling (will depend on child class for each environment
    # default tiling is for 10 state random walk
    def construct_features(self,states):
        # print(states)
        # can't quite understand the many hot encoding of the julia code
        # going to assume it is a one hot encoding of each coordinate, so [[0001000],[0100000]]
        # flattens encoding into a 1D vector: [[00010000100000]]
        return np.array([[np.float(i == s[0]) for i in range(11)] + [np.float(i == s[1]) for i in range(11)] for s in states])

    def value_function(self):
        test_states = [[i,j] for i in range(11) for j in range(11)]
        test_features = self.construct_features(test_states)
        values = np.reshape(self.model.predict([test_features, np.array([0.]*121)]), (11,11))
        values = values*np.array(np.invert(self.env.rooms), dtype=float) # zero the walls
        return values

    
        