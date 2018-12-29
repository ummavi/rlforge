# RLForge

RLForge is a lightweight framework to *quickly* and *cleanly* implement popular RL algorithms from literature as well as prototype new ones. Algorithms in RLForge are implemented using TensorFlow Eager but should be very easily modified to use libraries like PyTorch or Chainer. 


This framework is inspired by [ChainerRL](https://github.com/chainer/chainerrl) and [OpenAI baselines](https://github.com/openai/baselines) and is an attempt to modularize the many components of an RL system so they can be combined in a simple, readable way without excessive code duplication. This is achieved through the use of [Mixins](https://en.wikipedia.org/wiki/Mixin) where each "feature" is implemented in its own encapsulated way and is automatically used by the agent as soon as it's added as a base class.


## Getting Started
Mixins are integrated into the agent through the use of step_hooks and episode_hooks which are called before and after the agent interacts with the environment for a step or an episode respectively. An example of a hook is the Experience Replay which adds a replay buffer that stores transitions and is sampled from during the training process of an off-policy RL algorithm like a Q-Learning agent.

```python 
class ExperienceReplayMX(BaseMixin):
    def __init__(self, replay_buffer_size, minibatch_size):

        self.minibatch_size = minibatch_size
        self.replay_buffer = ReplayBuffer(replay_buffer_size)

        # This step controls how the agent uses the experience 
        # .. replay mixin. post_step_hooks are automatically
        # .. called after every step of the environment
        self.post_step_hooks.append(self.add_to_replay)

    def add_to_replay(self, global_step_ts, step_data):
        """
        post_step_hook Parameters:
            global_step_ts (int): Current global step counter
            step_data (tuple): A transition tuple (s,a,r,s',done)
        """
        s, a, r, s_n, d = step_data
        self.replay_buffer.append(s, a, r, s_n, d)

    def get_train_batch(self):
        """
        Overwrite the default get_train_batch to use a mini-batch from ER.
        """
        return self.replay_buffer.sample(self.minibatch_size)

```

A basic Q-Learning agent can be modified to use a replay buffer by simply modifying it's class definition from `class QLearningAgent(BaseAgent):` to 
`class QLearningAgent(ExperienceReplayMX, BaseAgent):` as below.

```python 
class QLearningAgent(ExperienceReplayMX, BaseAgent):

    def __init__(self, q_function, policy_learning_rate=0.001, gamma=0.8, 
                replay_buffer_size=5000, minibatch_size=32):

        BaseAgent.__init__(self, env)
        ExperienceReplayMX.__init__(self, replay_buffer_size, minibatch_size)

        self.model = q_function
        self.gamma = gamma
        
        self.opt = tf.train.AdamOptimizer(policy_learning_rate)

        #Q-Learning trains after every step so add it to the post_episode hook list
        self.post_episode_hooks.append(self.learn)


    def learn(self, global_step_ts , step_data):
        """Perform one step of learning with a batch of data

        post_step_hook Parameters:
            global_step_ts (int): Current global step counter
            step_data (tuple): A transition tuple (s,a,r,s',done)
        """
        states, actions, rewards, state_ns, dones = self.get_train_batch()
        rewards, dones = np.float32(rewards), np.float32(dones)

        q_star = tf.reduce_max(self.model(state_ns), axis=-1)
        q_target = rewards + (self.gamma*q_star*(1. - dones))

        with tf.GradientTape() as tape:
            preds = self.model(states)
            q_cur = tf.reduce_sum(preds * tf.one_hot(actions, self.env.n_actions), axis=-1)
            losses = tf.losses.huber_loss(labels=q_target, predictions=q_cur)
            grads = tape.gradient(losses, self.model.trainable_weights)
            self.opt.apply_gradients(zip(grads,self.model.trainable_weights))
```

## Roadmap
Whether this architecture choice holds up to more complex algorithms remains to be seen. Regardless, implementations of a number of standard Deep-RL algorithms followed by some multi-goal and multi-task algorithms is on the agenda.

### Core 
* Compatibility checks and dependencies for mixins
* More debugging tools, sanity checks and warnings
* Many more reusable mixins implementations! 

### Algorithms 
* Vanilla PG [+Baselines]
* Double & Duelling DQN
* C51 (and perhaps, it's successors)
* DDPG 
* PPO


