import numpy as np
import tensorflow as tf
from OffPolicyAgent_FourRooms import OffPolicyAgent_FourRooms
from prioritized_memory import Memory
from TransitionData import TransitionComponent, extract_transition_components
tf.compat.v1.disable_eager_execution()

# IMPORTANCE RESAMPLING agent class for Four Rooms env -
# derived from Off Policy agent class
class IRAgent_FourRooms(OffPolicyAgent_FourRooms):
    # construct agent's model separately, so it can be sized according to problem
    def __init__(self, n_replay, env, target_policy, behavior_policy, lr, discount, type='BC'):
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
        self.build_model(self.env.size[0]*self.env.size[1], 1)

    # instead of generating one episode of experience, take 16 steps of experience
    def generate_experience(self, k=16):

        # Initialize environment
        s = self.env.reset()
        done = False
        steps = 0 # counting to k steps

        while steps < k:

            # choose action according to policy
            a = np.random.choice(a=self.actions, p=self.behavior_policy[s[0],s[1]])

            # Take a step in environment based on chosen action
            (s2,r,done,_) = self.env.step(a)

            # Compute importance ratios
            ratio = self.target_policy[s[0],s[1],a] / self.behavior_policy[s[0],s[1],a]

            # Add experience to IR replay buffer
            self.replay_buffer.add(ratio, (s,a,r,s2))

            # Set for next step
            s=s2
            self.t += 1
            steps += 1

            # If episode ends, reset environment
            if done:
                s = self.env.reset()
                done = False

    # do batch of training using replay buffer
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
            # if experience ends in terminal state then s==s2
            if (states[i] ==  next_states[i]).all():
                next_values[i] = 0.0

        # Compute targets by bootstrap estimates
        targets = (rewards + self.discount*next_values)

        # Train on samples
        self.model.fit([state_features, ratios], targets, batch_size=batch_size, verbose=0)
