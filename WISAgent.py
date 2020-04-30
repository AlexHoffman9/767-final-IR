import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from OffPolicyAgent import OffPolicyAgent

tf.compat.v1.disable_eager_execution()

# Weighted Importance Sampling agent for Random Walk
class WISAgent(OffPolicyAgent):

    # construct agent's model separately, so it can be sized according to problem
    def __init__(self, n_replay, env, target_policy, behavior_policy, lr, discount, type='WIS_minibatch'):
        super().__init__(n_replay, env, target_policy, behavior_policy, lr, discount, type)

    def model_compile(self, ratios):
        # loss function for batch update
        # Weighted Importance Sampling loss function
        def wis_minibatch_loss(y_true,y_pred):
            ratio_sum = tf.reduce_sum(ratios)+.00000001
            se = tf.math.multiply(tf.math.square(y_true-y_pred), ratios)
            # weights loss according to sampling ratio. If ratio=0, sample is essentially ignored
            return tf.math.reduce_sum(se)/ratio_sum

        self.model.compile(loss = wis_minibatch_loss, optimizer = SGD(lr=self.lr))
