import numpy as np 
import tensorflow as tf
tf.enable_eager_execution()


from rlforge.agents.base_agent import BaseAgent

from rlforge.common.utils import Episode
from rlforge.mixins.policies import EpsilonGreedyPolicyMX
from rlforge.mixins.experience_replay import ExperienceReplayMX
from rlforge.mixins.target_network import TargetNetworkMX



class DQNAgent(EpsilonGreedyPolicyMX, ExperienceReplayMX, TargetNetworkMX, BaseAgent):
    """DQN
    """
    def __init__(self, env, q_fn, 
                policy_learning_rate, replay_buffer_size, target_network_update_freq,
                minibatch_size = 32, gamma=0.98, eps=0.2):

        self.model = q_fn
        self.gamma = gamma

        BaseAgent.__init__(self, env)
        ExperienceReplayMX.__init__(self, replay_buffer_size, minibatch_size)
        TargetNetworkMX.__init__(self, target_network_update_freq)
        EpsilonGreedyPolicyMX.__init__(self, eps_fixed=eps)

        self.opt = tf.train.AdamOptimizer(policy_learning_rate)

        #Since DQN Learns after every step.
        self.post_step_hooks.append(self.learn)

    def learn(self, global_step_ts, step_data):
        """Perform one step of learning with a batch of data

        post_step_hook Parameters:
            global_step_ts (int)
            step_data (tuple): (s,a,r,s',done)
        """
        states, actions, rewards, state_ns, dones = self.get_train_batch()
        rewards, dones = np.float32(rewards), np.float32(dones)

        q_star = tf.stop_gradient(tf.reduce_max(self.target_model(state_ns), axis=-1))
        q_target = rewards+(self.gamma*q_star*(1.-dones))

        with tf.GradientTape() as tape:
            preds = self.model(states)
            q_cur = tf.reduce_sum(preds * tf.one_hot(actions, self.env.n_actions), axis=-1)
            losses = tf.losses.huber_loss(labels=q_target, predictions=q_cur)
            grads = tape.gradient(losses, self.model.trainable_weights)
            self.opt.apply_gradients(zip(grads,
                                       self.model.trainable_weights))
        return losses 
    