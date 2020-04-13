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
import time


# Agent class with defaults for importance sampling
# can write new classes for WIS, IR, etc. by overwriting key functions
# Compatible with problem_name='RandomWalk' or 'FourRooms'
class OffPolicyAgent():
    # construct agent's model separately, so it can be sized according to problem
    def __init__(self, problem_name, n_replay, env, target_policy, behavior_policy, lr, discount):
        self.lr = lr
        self.discount = discount
        self.n_replay = n_replay
        self.env = env
        if problem_name == 'RandomWalk':
            self.actions = range(2)
            self.n_features = 10  # fixed for 10 state randomwalk. just using sparse coding
        elif problem_name == 'FourRooms':
            self.actions = range(4)
            self.n_features = 10 #TODO depends on tiling for states in fourroomsu. consult julia code. This needs to be input shape for the 2d tiled input
        else:
            print('ERRORRR: invalid problem name')
        self.model = self.build_model(self.n_features, 1)
        # print(self.model.summary())
        self.target_policy = target_policy # 2d array indexed by state, action. example: 10x2 array for 10 state random walk
        self.behavior_policy = behavior_policy
        self.replay_buffer = np.zeros(shape=(n_replay), dtype=[('s',np.int32), ('s2',np.int32), ('r',np.int32), ('ratio', np.float)]) # state, next state, reward, ratio
        self.t=0

    # build neural network for state value function
    # Default is single layer linear layer
    def build_model(self, input_dim, out_dim):
        input_layer = Input(shape=(input_dim), name='state_input')
        ratios = Input(shape=(1), name='importance_ratios')
        # hidden_layer = Dense(32, activation = "relu", name='hidden_layer')(input_layer)
        output_layer = Dense(out_dim, activation="linear", name='output_layer')(input_layer) #(hidden_layer)
        # loss function for batch update
        # just MSE loss multiplied by importance sampling ratio
        def is_loss(y_true, y_pred):
            se = tf.math.squared(y_true-y_pred) * ratios # weights loss according to sampling ratio. If ratio=0, sample is essentially ignored
            return tf.math.reduce_mean(se)
        # opt = Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, amsgrad=True)
        model = Model(inputs=[input_layer, ratios], outputs=[output_layer])
        model.compile(loss="mean_squared_error", optimizer = SGD(lr=self.lr))
        return model

    # complete episode of experience and then train using buffer
    # not sure how much experience to get before training on it...one episode? 2? n timesteps?
    def generate_episode(self):
        # init state
        s = self.env.reset()
        done = False
        while not done:
            # choose action according to policy
            a = np.random.choice(a=self.actions, p=self.behavior_policy[s])
            (s2,r,done,_) = env.step(a)
            ratio = self.target_policy[s,a] / self.behavior_policy[s,a]
            self.replay_buffer[self.t%self.n_replay] = (s,s2,r,ratio)
            s=s2
            self.t += 1

    # do batch of training using replay buffer
    # Default is to do a minibatch update. The paper uses both minibatch and incremental updates, so this could be changed
    def train_batch(self, n_samples, batch_size):
        sample_indices = self.sample_buffer(n_samples)
        # compute targets = ratio*(r + v(s'))
        rewards = self.replay_buffer['r'][sample_indices]
        # print("sample indices:", sample_indices)
        next_state_features = self.construct_features(self.replay_buffer['s2'][sample_indices])
        state_features = self.construct_features(self.replay_buffer['s'][sample_indices])
        ratios = self.replay_buffer['ratio'][sample_indices]
        next_values = self.model.predict(next_state_features).flatten()
        # v(s') is zero for terminal state, so need to fix model prediction
        for i in range(n_samples):
            # if experience ends in terminal state, value function returns 0
            if self.replay_buffer['s2'][sample_indices[i]] == -1 or self.replay_buffer['s2'][sample_indices[i]] == 10: #TODO this only works for randomwalk of size 10

                next_values[i] = 0.0
        # targets = (rewards + self.discount*next_values)*ratios # this was wrong. the weight update is multiplied by the sampling ratio, not the target
        targets = (rewards + self.discount*next_values)
        # testing the targets for the last state because its value is off
        # for i in range(n_samples):
        #     if self.replay_buffer['s'][sample_indices[i]] == 9:
        #         print(targets[i])
        # print("state features:", state_features)
        # print('rewards:', rewards)
        # print("ratios", ratios)
        # print('next values:', next_values)
        # print('targets',targets)
        # train on samples
        self.model.fit([state_features, ratios], targets, batch_size=batch_size, verbose=0)

    # choose batch of experience from buffer. IR will change this function
    # default: random n samples
    def sample_buffer(self, n_samples):
        max_index = self.n_replay
        if self.t < self.n_replay:
            max_index = self.t
        return np.random.randint(low=0, high=max_index, size=n_samples, dtype=int)
        # return [self.buffer_index-n_samples+i for i in range(n_samples)]

    # need to choose tiling (will depend on child class for each environment
    # default tiling is for 10 state random walk
    def construct_features(self,states):
        # print(states)
        return np.array([[np.float(i == s) for i in range(self.n_features)] for s in states])



