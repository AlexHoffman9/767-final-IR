import numpy as np
import gym
import tensorflow as tf
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from random_walk_env import RandomWalkEnv
from four_rooms_env import FourRoomsEnv
import time
from OffPolicyAgent_FourRooms import OffPolicyAgent_FourRooms
from prioritized_memory import Memory
from TransitionData import TransitionComponent, extract_transition_components
tf.compat.v1.disable_eager_execution()


# get(dict,key) returns value for key


# Agent class with defaults for importance sampling
# can write new classes for WIS, IR, etc. by overwriting key functions
# Compatible with problem_name='RandomWalk' or 'FourRooms'
class IRAgent_FourRooms(OffPolicyAgent_FourRooms):
    # construct agent's model separately, so it can be sized according to problem
    def __init__(self, n_replay, env, target_policy, behavior_policy, lr, discount, IS_method='BC'):
        super().__init__(n_replay, env, target_policy, behavior_policy, lr, discount, IS_method)

        # IR Replay buffer
        self.replay_buffer = Memory(n_replay)

    # reseed numpy, reset weights of network
    def reset(self,seed):
        self.t=0
        np.random.seed(seed)
        self.replay_buffer = Memory(self.n_replay)
        self.build_model(self.n_features*2, 1, self.name)

    def model_compile(self, ratios, IS_method):
        # loss function for batch update
        # just MSE loss multiplied by importance sampling ratio
        def ir_loss(y_true, y_pred):
            se = tf.math.multiply(tf.math.square(y_true-y_pred), ratios) # weights loss according to sampling ratio. If ratio=0, sample is essentially ignored
            return tf.math.reduce_mean(se)

        # Compile model from loaded
        self.model.compile(loss=ir_loss, optimizer = SGD(lr=self.lr))

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

            # IR Replay add
            # self.replay_buffer[self.t%self.n_replay] = (s,s2,r,ratio)
            self.replay_buffer.add(ratio, (s,a,r,s2))
            s=s2
            self.t += 1
            steps += 1
            if done:
                s = self.env.reset()
                done = False

    # do batch of training using replay buffer
    # Default is to do a minibatch update. The paper uses both minibatch and incremental updates, so this could be changed
    def train_batch(self, n_samples, batch_size):

        # IR Sample
        data_samples, _, priorities, buffer_total = self.replay_buffer.sample(n_samples)
        # compute targets = ratio*(r + v(s'))

        # IR extract components
        rewards = extract_transition_components(data_samples, TransitionComponent.reward)
        next_states = extract_transition_components(data_samples, TransitionComponent.next_state)
        next_state_features = self.construct_features(next_states)
        states = extract_transition_components(data_samples, TransitionComponent.state)
        state_features = self.construct_features(states)

        # Dummy ratios - Bias correction
        ratios = np.ones(len(states))

        if self.name == "BC":
            ratios = ratios*(buffer_total/self.replay_buffer.tree.n_entries)


        # ratios = self.replay_buffer['ratio'][sample_indices]
        next_values = self.model.predict([next_state_features, np.zeros(next_state_features.shape[0])]).flatten()
        # v(s') is zero for terminal state, so need to fix model prediction
        for i in range(n_samples):
            # if experience ends in terminal state then s==s2
            if (states[i] ==  next_states[i]).all():
                next_values[i] = 0.0
        targets = (rewards + self.discount*next_values)

        # Train on samples
        return self.model.fit([state_features, ratios], targets, batch_size=batch_size, verbose=0) #, callbacks=[TerminateOnNaN()])
