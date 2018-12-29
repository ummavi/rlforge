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

    def learn(self, global_step_ts , step_data):
        """Perform one step of learning with a batch of data

        post_step_hook Parameters:
            global_step_ts (int)
            step_data (tuple): (s,a,r,s',done)
        """
        states, actions, rewards, state_ns, dones = self.get_train_batch()
        rewards, is_not_terminal = np.float32(rewards), np.float32(np.invert(dones))

        q_star = tf.stop_gradient(tf.reduce_max(self.target_model(state_ns), axis=-1))
        q_target = rewards+(self.gamma*q_star*is_not_terminal)

        with tf.GradientTape() as tape:
            preds = self.model(states)
            q_cur = tf.reduce_sum(preds * tf.one_hot(actions, self.env.n_actions), axis=-1)
            losses = tf.losses.huber_loss(labels=q_target, predictions=q_cur)
            grads = tape.gradient(losses, self.model.trainable_weights)
            self.opt.apply_gradients(zip(grads,
                                       self.model.trainable_weights))
        
        self.stats.append("step_losses",global_step_ts,losses)
        self.stats.append("step_mean_q",global_step_ts,np.mean(preds))



class DoubleDQNAgent(DQNAgent):
    """Double Deep-Q Learning Agent

    Implementation of the Double DQN algorithm 
    (https://arxiv.org/abs/1509.06461)


    DQN Targets:
    Y =  r + gamma * max_a(Q_target(s_new,a))

    DoubleDQN Targets:
    Y = r + gamma * Q_target(s_new, argmax_a(Q(s_new,a)))

    """
    def learn(self, global_step_ts , step_data):
        """Perform one step of learning with a batch of data

        post_step_hook Parameters:
            global_step_ts (int)
            step_data (tuple): (s,a,r,s',done)
        """
        states, actions, rewards, state_ns, dones = self.get_train_batch()
        rewards, is_not_terminal = np.float32(rewards), np.float32(np.invert(dones))

        best_sn_actions = np.argmax(self.model(state_ns), axis=-1)

        target_preds = self.target_model(state_ns)
        target_preds = tf.reduce_sum(target_preds * tf.one_hot(best_sn_actions, self.env.n_actions), axis=-1)


        q_target = rewards+(self.gamma*target_preds*is_not_terminal)

    

        with tf.GradientTape() as tape:
            preds = self.model(states)
            q_cur = tf.reduce_sum(preds * tf.one_hot(actions, self.env.n_actions), axis=-1)
            losses = tf.losses.huber_loss(labels=q_target, predictions=q_cur)
            grads = tape.gradient(losses, self.model.trainable_weights)
            self.opt.apply_gradients(zip(grads,
                                       self.model.trainable_weights))
        
        self.stats.append("step_losses",global_step_ts,losses)
        self.stats.append("step_mean_q",global_step_ts,np.mean(preds))



if __name__ == '__main__':
    from tqdm import tqdm
    from rlforge.common.q_functions import QNetwork
    from rlforge.environments.environment import GymEnv

    def train(agent, n_episodes):
        # Simple train function using tqdm to show progress
        pbar = tqdm(range(n_episodes))
        for i in pbar:
            e = agent.interact(1)
            if i%50==0:
                ts,last_50_rets = map(list, zip(*agent.stats.get("episode_returns")[-50:])) 
                pbar.set_description("Latest return: "+str(np.mean(last_50_rets)))

    np.random.seed(0)
    tf.set_random_seed(0)   
    env = GymEnv('CartPole-v0')
    q_network = QNetwork(env.n_actions,env.d_observations,dict(layer_sizes=[24,24],
                        activation="relu"))
    agent = DQNAgent(env, q_network,policy_learning_rate=0.005,
                    target_network_update_freq=200, 
                    replay_buffer_size=5000,
                    gamma=0.8,
                    eps=0.2,
                    minibatch_size=64)
    train(agent,2500)
