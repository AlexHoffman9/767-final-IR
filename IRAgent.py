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
from prioritized_memory import Memory
from TransitionData import TransitionComponent, extract_transition_components
from OffPolicyAgent import OffPolicyAgent

tf.compat.v1.disable_eager_execution()

# Agent class with defaults for importance sampling
# can write new classes for WIS, IR, etc. by overwriting key functions
# Compatible with problem_name='RandomWalk' or 'FourRooms'
class IRAgent(OffPolicyAgent):
    # construct agent's model separately, so it can be sized according to problem
    def __init__(self, problem_name, n_replay, env, target_policy, behavior_policy, lr, discount, batch_corr = True):
        self.batch_corr = batch_corr
        super().__init__(problem_name, n_replay, env, target_policy, behavior_policy, lr, discount)
        # IR Replay buffer
        self.replay_buffer = Memory(n_replay)
        self.name = 'IR'

    # complete episode of experience and then train using buffer
    # not sure how much experience to get before training on it...one episode? 2? n timesteps?
    def generate_episode(self, k=16):
        # init state
        s = self.env.reset()
        done = False
        steps = 0
        while steps < k:
            # choose action according to policy
            a = np.random.choice(a=self.actions, p=self.behavior_policy[s])
            (s2,r,done,_) = self.env.step(a)
            ratio = self.target_policy[s,a] / self.behavior_policy[s,a]

            # IR Replay add
            self.replay_buffer.add(ratio, (s,a,r,s2))
            #self.replay_buffer[self.t%self.n_replay] = (s,s2,r,ratio)
            s=s2
            self.t += 1
            steps += 1
            if done:
                done = False
                s = self.env.reset()

    # do batch of training using replay buffer
    # Default is to do a minibatch update. The paper uses both minibatch and incremental updates, so this could be changed
    def train_batch(self, n_samples, batch_size):

        # IR Sample
        replay_samples = self.replay_buffer.sample(n_samples)
        data_samples = replay_samples[0]
        buffer_total = replay_samples[3]

        rewards = extract_transition_components(data_samples, TransitionComponent.reward)
        next_states = extract_transition_components(data_samples, TransitionComponent.next_state)
        next_state_features = self.construct_features(next_states)
        states = extract_transition_components(data_samples, TransitionComponent.state)
        state_features = self.construct_features(states)
        #print(len(data_samples), len(next_states), n_samples)

        # print(buffer_total)

        # Dummy ratios - Batch Correction
        ratios = np.ones(len(states))

        if self.batch_corr:
            ratios = ratios*(buffer_total/self.replay_buffer.tree.n_entries)

        next_values = self.model.predict([next_state_features, np.zeros(next_state_features.shape[0])]).flatten()
        # v(s') is zero for terminal state, so need to fix model prediction
        for i in range(n_samples):
            # if experience ends in terminal state, value function returns 0
            if next_states[i] == -1 or next_states[i] == 10: #TODO this only works for randomwalk of size 10
                next_values[i] = 0.0
        # targets = (rewards + self.discount*next_values)*ratios # this was wrong. the weight update is multiplied by the sampling ratio, not the target
        targets = (rewards + self.discount*next_values)

        # train on samples
        self.model.fit([state_features, ratios], targets, batch_size=batch_size, verbose=0)

    def value_function(self):
        states = self.construct_features(range(10))
        values = self.model.predict([states, np.array([0.]*10)])
        return values
