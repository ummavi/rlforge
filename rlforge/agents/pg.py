import numpy as np 
import tensorflow as tf

tf.enable_eager_execution()

from rlforge.common.utils import Episode
from rlforge.agents.base_agent import BaseAgent
from rlforge.mixins.policies import SoftmaxPolicyMX


class REINFORCEAgent(SoftmaxPolicyMX,BaseAgent):
    """
    Simple Episodic REINFORCE (Williams92) agent with discrete actions
    and a softmax policy.

    TODO: Entropy regularization coefficient, Baselines
    """
    def __init__(self, env, model, policy_learning_rate, gamma=0.9):
        """
        Parameters:
        model: A callable model with the final layer being identity.
        """
        BaseAgent.__init__(self,env)
        SoftmaxPolicyMX.__init__(self)

        #Using a discrete softmax policy.
        self.model = model
        self.gamma = gamma 
        self.opt = tf.train.AdamOptimizer(policy_learning_rate)

        self.post_episode_hooks.append(self.learn)

    def learn(self, global_episode_ts , episode_data):
        """Perform one step of learning with a batch of data

        post_step_hook Parameters:
            global_step_ts (int)
            episode_data (Episode)
        """

        discount_multipliers = np.power(self.gamma,np.arange(episode_data.length))
        discounted_returns = []
        for i in range(len(episode_data.rewards)):
            future_rewards = np.float32(episode_data.rewards[i:])
            discount_multipliers = discount_multipliers[:len(future_rewards)]
            discounted_return = np.sum(future_rewards*discount_multipliers)
            discounted_returns.append(discounted_return)
        discounted_returns = np.float32(discounted_returns)

        states, actions = episode_data.observations[:-1], episode_data.actions

        with tf.GradientTape() as tape:
            probs = tf.nn.softmax(self.model(states))
            logprobs = tf.log(probs)

            selected_logprobs = tf.reduce_sum(logprobs*tf.one_hot(actions,
                                         self.env.n_actions),axis=-1)


            losses = -tf.reduce_sum(selected_logprobs*discounted_returns) 
            grads = tape.gradient(losses, self.model.trainable_weights)
            self.opt.apply_gradients(zip(grads,
                                       self.model.trainable_weights))        

        self.stats.append("episode_losses",global_episode_ts,losses)



if __name__ == '__main__':
    from tqdm import tqdm
    from rlforge.common.policy_functions import PolicyNetworkDense
    from rlforge.environments.environment import GymEnv

    def train(agent, n_episodes):
        # Simple train function using tqdm to show progress
        pbar = tqdm(range(n_episodes))
        for i in pbar:
            e = agent.interact(1)
            if i%5==0:
                last_5_rets = agent.stats.get_values("episode_returns")[-5:]
                pbar.set_description("Latest return: "+str(np.mean(last_5_rets)))

    env = GymEnv('CartPole-v0')

    np.random.seed(0)
    tf.set_random_seed(0)
    env.env.seed(0)

    policy_network = PolicyNetworkDense(env.n_actions,dict(layer_sizes=[64,64],
                                                activation="tanh"))
    agent = REINFORCEAgent(env, policy_network,
                        policy_learning_rate=0.001,
                        gamma=0.99)
    train(agent,1000)
    print("Average Return (Train)",np.mean(agent.stats.get_values("episode_returns")))
