import numpy as np 
import tensorflow as tf
tf.enable_eager_execution()

from rlforge.common.utils import Episode
from rlforge.agents.base_agent import BaseAgent
from rlforge.mixins.policies import EpsilonGreedyPolicyMX
from rlforge.mixins.experience_replay import ExperienceReplayMX
from rlforge.mixins.target_network import TargetNetworkMX



class DQNAgent(EpsilonGreedyPolicyMX, ExperienceReplayMX, 
                TargetNetworkMX, BaseAgent):
    """Deep-Q Network agent

    Implementation of the DQN agent as described in the nature paper
    (https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)


    * Huber Loss
    * Epsilon-greedy Policy (EpsilonGreedyPolicyMX)
    * Experience Replay (ExperienceReplayMX)
    * Uses Target Network to compute Q* for the target (TargetNetworkMX)


    Note: Python inheritance from R-L. List BaseAgent at the end
    """
    def __init__(self, env, q_function, policy_learning_rate, 
                replay_buffer_size, target_network_update_freq,
                minibatch_size = 32, gamma=0.98, eps=0.2):

        self.model = q_function
        self.gamma = gamma

        BaseAgent.__init__(self, env)
        ExperienceReplayMX.__init__(self, replay_buffer_size, minibatch_size)
        TargetNetworkMX.__init__(self, target_network_update_freq)
        EpsilonGreedyPolicyMX.__init__(self, eps_fixed=eps)

        self.opt = tf.train.AdamOptimizer(policy_learning_rate)

        #DQN trains after every step so add it to the post_episode hook
        self.post_episode_hooks.append(self.learn)

    def learn(self, global_step_ts:int , step_data:tuple):
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

if __name__ == '__main__':
    from tqdm import tqdm
    from rlforge.common.q_functions import QNetwork
    from rlforge.environments.environment import Gym_env

    def train(agent, n_episodes):
        train_returns = []
        pbar = tqdm(range(n_episodes))
        for i in pbar:
            e = agent.interact(1)
            train_returns.append(np.sum(e[0].rewards))
            pbar.set_description("Latest return:"+str(train_returns[-1]))
        return train_returns

    np.random.seed(0)
    tf.set_random_seed(0)   
    env = Gym_env('CartPole-v0')
    q_network = QNetwork(env.n_actions,env.d_observations,dict(layer_sizes=[24,24],
                        activation="relu"))
    agent = DQNAgent(env, q_network,policy_learning_rate=0.005,
                    target_network_update_freq=200, 
                    replay_buffer_size=5000,
                    gamma=0.8,
                    eps=0.2,
                    minibatch_size=128)
    train(agent,2500)
