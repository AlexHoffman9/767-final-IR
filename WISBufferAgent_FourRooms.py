import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from OffPolicyAgent_FourRooms import OffPolicyAgent_FourRooms
tf.compat.v1.disable_eager_execution()

# Weighted Importance Sampling Minibatch Agent for Four Rooms env
class WISMinibatchAgent_FourRooms(OffPolicyAgent_FourRooms):

    # construct agent's model separately, so it can be sized according to problem
    def __init__(self, n_replay, env, target_policy, behavior_policy, lr, discount, type='IS'):
        super().__init__(n_replay, env, target_policy, behavior_policy, lr, discount, type)

    def model_compile(self, ratios):
        # loss function for batch update
        def wis_buffer_loss(y_true,y_pred):
            buffer_entries = np.min(self.t,self.n_replay)
            ratio_sum = np.sum(self.replay_buffer['ratio'][0:buffer_entries]) # only sum entries in buffer up to current size of buffer
            k = len(ratios)
            se = tf.math.multiply(tf.math.square(y_true-y_pred), ratios) # weights loss according to sampling ratio. If ratio=0, sample is essentially ignored
            return buffer_entries*tf.math.reduce_sum(se)/(ratio_sum*k) # n/k * (sum errors / sum ratios)

        # Compile current model
        self.model.compile(loss=wis_buffer_loss, optimizer = SGD(lr=self.lr))
