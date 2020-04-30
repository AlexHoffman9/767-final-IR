import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from prioritized_memory import Memory
from TransitionData import TransitionComponent, extract_transition_components
from OffPolicyAgent import OffPolicyAgent

# Prevent Eager execution from tensorflow
tf.compat.v1.disable_eager_execution()

# IMPORTANCE RESAMPLING agent class for Random Walk -
# derived from Off Policy agent class
class IRAgent(OffPolicyAgent):
    # construct agent's model separately, so it can be sized according to problem
    # Constructor arguments - type == 'BC' indicates Bias correction
    def __init__(self, n_replay, env, target_policy, behavior_policy, lr, discount, type = 'BC'):
        super().__init__(n_replay, env, target_policy, behavior_policy, lr, discount, type)

    # reseed numpy, reset weights of network
    # Reset must be performed before every episode
    def reset(self,seed=0):
        # Reset time
        self.t=0

        # Set seed value
        np.random.seed(seed)

        # Reset replay buffer
        self.replay_buffer = Memory(self.n_replay)

        # Rebuild model
        self.build_model(self.n_features,1)

    # Generate steps of experience
    def generate_experience(self, k=16):

        # Initialize environment
        s = self.env.reset()
        done = False
        steps = 0

        # For each step
        while steps < k:

            # choose action according to policy
            a = np.random.choice(a=self.actions, p=self.behavior_policy[s])

            # Take a step in environment based on chosen action
            (s2,r,done,_) = self.env.step(a)

            # Compute importance ratios
            ratio = self.target_policy[s,a] / self.behavior_policy[s,a]

            # Add experience to IR replay buffer
            self.replay_buffer.add(ratio, (s,a,r,s2))

            # Set for next step
            s=s2
            self.t += 1
            steps += 1

            # If episode ends, reset environment
            if done:
                done = False
                s = self.env.reset()

    # Do batch of training using replay buffer
    def train_batch(self, batch_size):

        # Sample a minibatch from replay buffer
        data_samples, _, _, buffer_total = self.replay_buffer.sample(batch_size)

        # Extract rewards, states, next states from samples
        rewards = extract_transition_components(data_samples, TransitionComponent.reward)
        next_states = extract_transition_components(data_samples, TransitionComponent.next_state)
        next_state_features = self.construct_features(next_states)
        states = extract_transition_components(data_samples, TransitionComponent.state)
        state_features = self.construct_features(states)

        # Importance ratios for update equation - IR does not use this
        ratios = np.ones(len(states))

        # In case of Bias Correction, pre-multiply bias corrector to update
        if self.name == "BC":
            ratios = ratios*(buffer_total/self.replay_buffer.tree.n_entries)

        # Get value estimate for next state
        next_values = self.model.predict([next_state_features, np.zeros(next_state_features.shape[0])]).flatten()

        # v(s') is zero for terminal state, so need to fix model prediction
        for i in range(batch_size):
            # if experience ends in terminal state, value function returns 0
            if next_states[i] == -1 or next_states[i] == 10: #TODO this only works for randomwalk of size 10
                next_values[i] = 0.0

        # Compute targets by bootstrap estimates
        targets = (rewards + self.discount*next_values)

        # train on samples
        self.model.fit([state_features, ratios], targets, batch_size=batch_size, verbose=0)
