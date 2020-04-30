import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from prioritized_memory import Memory
from TransitionData import TransitionComponent, extract_transition_components
from OffPolicyAgent import OffPolicyAgent

tf.compat.v1.disable_eager_execution()

# Prioritized Experience Replay Agent for Random Walk
class PERAgent(OffPolicyAgent):

    # construct agent's model separately, so it can be sized according to problem
    def __init__(self, n_replay, env, target_policy, behavior_policy, lr, discount, type = 'BC'):
        super().__init__(n_replay, env, target_policy, behavior_policy, lr, discount, type)

    # reseed numpy, reset weights of network
    # Reset must be performed before every episode
    def reset(self,seed):
        # Reset time
        self.t=0

        # Set seed value
        np.random.seed(seed)

        # Reset replay buffer
        self.replay_buffer = Memory(self.n_replay)

        # Rebuild model
        self.build_model(self.n_features,self.env.nA)
        
    def generate_action(self, s, target_policy_sel = True):
        pval = self.target_policy[s] if target_policy_sel else self.behavior_policy[s]
        return np.random.choice(a=self.actions, p=pval)

    def generate_all_actions(self,target_policy_sel = True):
        return np.array([self.generate_action(item, target_policy_sel) for item in range(self.target_policy.shape[0])])

    # Generate steps of experience
    def generate_experience(self, k=16):

        # Initialize environment
        s = self.env.reset()
        done = False
        steps = 0

        # For each step
        while steps < k:

            # choose action according to behavior policy
            a = self.generate_action(s,False)

            # Take a step in environment based on chosen action
            (s2,r,done,_) = self.env.step(a)

            # Compute importance ratios
            ratio = self.target_policy[s,a] / self.behavior_policy[s,a]

            # states and target action for Computing TD Error
            current_state = self.construct_features([s])
            next_state = self.construct_features([s2])
            target_policy_action = self.generate_action(s,True)

            # Get bootstrap estimate of next state action values
            value_s = self.model.predict([current_state,np.zeros(current_state.shape[0])])
            value_next_s = self.model.predict([next_state,np.zeros(next_state.shape[0])])
            updated_val = r if done else (r + self.discount*value_next_s[0][target_policy_action])

            # Compute TD error
            td_error = np.abs(updated_val - value_s[0][a])

            # Stop execution if weights blow up - not converged
            if td_error > 10**5:
                return 1

            # Add experience to IR replay buffer
            self.replay_buffer.add_per(td_error, (s,a,r,s2))

            # Set for next step
            s=s2
            self.t += 1
            steps += 1

            # If episode ends, reset environment
            if done:
                done = False
                s = self.env.reset()
        return 0

    # do batch of training using replay buffer
    def train_batch(self, n_samples, batch_size):

        # Sample a minibatch from replay buffer
        data_samples, idxs, ratios, buffer_total = self.replay_buffer.sample(n_samples)

        # Extract rewards, states, next states, actions from samples
        rewards = extract_transition_components(data_samples, TransitionComponent.reward)
        next_states = extract_transition_components(data_samples, TransitionComponent.next_state)
        next_state_features = self.construct_features(next_states)
        states = extract_transition_components(data_samples, TransitionComponent.state)
        state_features = self.construct_features(states)
        actions = extract_transition_components(data_samples, TransitionComponent.action)

        # Calculate Target policy actions
        target_policy_actions = np.array([self.generate_action(state, True) for state in states])

        # Calculate state values for TD error
        next_values_sa = self.model.predict([next_state_features, np.zeros(next_state_features.shape[0])])
        next_values = np.choose(target_policy_actions,next_values_sa.T)

        # v(s') is zero for terminal state, so need to fix model prediction
        for i in range(n_samples):
            # if experience ends in terminal state, value function returns 0
            if next_states[i] == -1 or next_states[i] == 10: #TODO this only works for randomwalk of size 10
                next_values[i] = 0.0

        # Compute targets by bootstrap estimates
        targets = (rewards + self.discount*next_values)

        # Compute error for updating priorities
        pred_values = self.model.predict([state_features, np.zeros(state_features.shape[0])])
        final_targets = np.copy(pred_values)
        np.put_along_axis(final_targets, np.expand_dims(actions,axis = 1),targets[:,np.newaxis],axis = 1)
        pred = np.choose(actions, pred_values.T)
        error = np.abs(pred - targets)

        # Priority update
        for i in range(batch_size):
            self.replay_buffer.update(idxs[i], error[i])

        # train on samples
        self.model.fit([state_features, ratios], final_targets, batch_size=batch_size, verbose=0)
