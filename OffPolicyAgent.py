import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
# Prevent Eager execution from tensorflow
tf.compat.v1.disable_eager_execution()

# Off Policy Agent class with defaults for Importance Sampling
# This forms the base class for rest of the Agents
class OffPolicyAgent():

    # construct agent's model separately, so it can be sized according to problem
    # Constructor arguments - type used for indicating agent information
    def __init__(self, n_replay, env, target_policy, behavior_policy, lr, discount, type='IS'):

        # Learning rate
        self.lr = lr

        # Discount factor
        self.discount = discount

        # Size of replay buffer
        self.n_replay = n_replay

        # Environment used for the interactions
        self.env = env

        # Number of actions
        self.actions = range(self.env.nA)

        # Number of features per state
        self.n_features = self.env.size  # 10 features for 10 state randomwalk. just using sparse coding

        # Target and Behavior policies to be evaluated
        self.target_policy = target_policy # 2d array indexed by state, action. example: 10x2 array for 10 state random walk
        self.behavior_policy = behavior_policy

        # Agent information for plots
        self.name = type

    # reseed numpy, reset weights of network
    # Reset must be performed before every episode
    def reset(self,seed=0):

        # Reset time
        self.t=0

        # Set seed value
        np.random.seed(seed)

        # Reset replay buffer
        self.replay_buffer = np.zeros(shape=(self.n_replay), dtype=[('s',np.int32), ('s2',np.int32), ('r',np.int32), ('ratio', np.float)]) # state, next state, reward, ratio

        # Rebuild model
        self.build_model(self.n_features,1)

    # Assign loss function to model and compile
    def model_compile(self, mse_weights):

        # loss function for batch update
        # just MSE loss multiplied by weight (Importance sampling ratio)
        def weighted_mse_loss(y_true, y_pred):
            se = tf.math.multiply(tf.math.square(y_true-y_pred), mse_weights) # weights loss according to sampling ratio. If ratio=0, sample is essentially ignored
            return tf.math.reduce_mean(se)

        self.model.compile(loss = weighted_mse_loss, optimizer = SGD(lr=self.lr))

    # build neural network for state value function
    # Default is single layer linear layer
    def build_model(self, input_dim, out_dim):

        # Input features
        input_layer = Input(shape=(input_dim), name='state_input')

        # MSE weights (Importance Ratios) converted to Tensor form
        mse_weights = Input(shape=(1), name='mse_weights')

        # Linear combination - output
        output_layer = Dense(out_dim, activation="linear", name='output_layer')(input_layer)#, kernel_initializer=tf.keras.initializers.Zeros(),
        self.model = Model(inputs=[input_layer, mse_weights], outputs=[output_layer])

        # Compile the model by adding loss function
        self.model_compile(mse_weights)

    # Generate steps of experience
    def generate_experience(self, k=16):

        # Initialize environment
        s = self.env.reset()
        done = False
        steps = 0

        # For each step
        while steps < k:

            # choose action according to behavior policy
            a = np.random.choice(a=self.actions, p=self.behavior_policy[s])

            # Take a step in environment based on chosen action
            (s2,r,done,_) = self.env.step(a)

            # Compute importance ratios
            ratio = self.target_policy[s,a] / self.behavior_policy[s,a]

            # Add experience to IS replay buffer
            self.replay_buffer[self.t%self.n_replay] = (s,s2,r,ratio)

            # Set for next step
            s=s2
            self.t += 1
            steps += 1

            # If episode ends, reset environment
            if done:
                done = False
                s = self.env.reset()

    # check if terminal state
    def check_terminal(self, sample):
        if self.replay_buffer['s2'][sample] == -1 or self.replay_buffer['s2'][sample] == self.env.size:
            return True
        else:
            return False
    # Do batch of training using replay buffer
    def train_batch(self, batch_size):

        # Sample a minibatch from replay buffer
        sample_indices = self.sample_buffer(batch_size)

        # Extract rewards, states, next states and ratios from samples
        rewards = self.replay_buffer['r'][sample_indices]
        next_state_features = self.construct_features(self.replay_buffer['s2'][sample_indices])
        state_features = self.construct_features(self.replay_buffer['s'][sample_indices])
        ratios = self.replay_buffer['ratio'][sample_indices]

        # Get value estimate for next state
        next_values = self.model.predict([next_state_features, np.zeros(next_state_features.shape[0])]).flatten()

        # v(s') is zero for terminal state, so need to fix model prediction
        for i in range(batch_size):

            # if experience ends in terminal state then s==s2
            if self.check_terminal(sample_indices[i]):
                next_values[i] = 0.0

        # Compute targets by bootstrap estimates
        targets = (rewards + self.discount*next_values)

        # train on samples
        self.model.fit([state_features, ratios], targets, batch_size=batch_size, verbose=0)

    # choose batch of experience from buffer.
    # default: random n samples
    def sample_buffer(self, n_samples):

        # Select buffer size according to t
        max_index = self.t if self.t < self.n_replay else self.n_replay

        # Uniformly randomly sample from buffer
        return np.random.randint(low=0, high=max_index, size=n_samples, dtype=int)

    # need to choose tiling (will depend on child class for each environment
    # default tiling is for 10 state random walk
    def construct_features(self,states):
        return np.array([[np.float(i == s) for i in range(self.n_features)] for s in states])

    # Return the entire value function estimate
    def value_function(self):
        states = self.construct_features(range(10))
        values = self.model.predict([states, np.array([0.]*10)])
        return values
