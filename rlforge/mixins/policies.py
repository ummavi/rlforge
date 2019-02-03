import numpy as np
import tensorflow as tf

from rlforge.mixins.base_mixin import BaseMixin


class EpsilonGreedyPolicyMX(BaseMixin):
    """Epsilon-Greedy policy with or w/o an epslion schedule
    """

    def __init__(self,
                 eps_fixed=0.2,
                 episodic_update=False,
                 eps_start=None,
                 eps_end=None,
                 ts_eps_end=None,
                 eps_schedule=None):
        """
        Parameters:
        eps_schedule {None | "linear"}: Chooses between a fixed learning
                                         rate and a decay schedule
        """
        assert eps_schedule in [None, "linear"]
        BaseMixin.__init__(self)

        self.eps_schedule = eps_schedule

        if eps_schedule is None:
            self.eps_start = self.eps_end = eps_fixed
            self.eps_delta = 0.0

        self.ts_eps = 0  # Note, this ts might be different from the agent's.

        if eps_schedule == "linear":
            assert eps_start is not None
            assert eps_end is not None
            assert ts_eps_end is not None

            self.eps_delta = float(eps_start - eps_end) / ts_eps_end
            self.eps_end = eps_end
            self.eps_start = eps_start
            self.ts_eps_end = ts_eps_end

        if episodic_update:
            self.post_episode_hooks.append(self.update_eps)
        else:
            self.post_step_hooks.append(self.update_eps)

    def act(self, state, greedy):
        """Epsilon greedy policy:
        """
        eps_current = max((self.eps_start - self.ts_eps * self.eps_delta),
                          self.eps_end)
        if greedy or np.random.uniform() > eps_current:
            q_values = self.network([state])[0]
            return np.argmax(q_values)
        else:
            return np.random.choice(self.env.n_actions)

    def update_eps(self, global_any_ts, any_data):
        """Updates the eps-greedy timestep.
        This can be either a post-episode or a post-step ook.
        """
        self.ts_eps += 1


class SoftmaxPolicyMX(BaseMixin):
    """Softmax policy.
    """

    def __init__(self):
        BaseMixin.__init__(self)

    def act(self, state, greedy):
        """Epsilon greedy policy:
        """
        numerical_prefs = self.model([state])
        policy_probs = np.float32(self.all_probs(numerical_prefs)[0])
        if greedy:
            return np.argmax(policy_probs)
        else:
            return np.random.choice(self.env.n_actions, p=policy_probs)

    def all_probs(self, numerical_prefs):
        return tf.nn.softmax(numerical_prefs)

    def all_logprobs(self, numerical_prefs):
        """Get log probabilities of policy.
        """
        return tf.nn.log_softmax(numerical_prefs)

    def probs(self, numerical_prefs, actions):
        all_probs = self.all_probs(numerical_prefs)
        selected_probs = all_probs * \
                         tf.one_hot(actions, self.env.n_actions)
        selected_probs = tf.reduce_sum(selected_probs, axis=-1)
        return selected_probs

    def logprobs(self, numerical_prefs, actions):
        all_logprobs = self.all_logprobs(numerical_prefs)
        selected_logprobs = all_logprobs * \
                tf.one_hot(actions, self.env.n_actions)
        selected_logprobs = tf.reduce_sum(selected_logprobs, axis=-1)
        return selected_logprobs

    def policy_entropy(self, numerical_prefs):
        probs = self.all_probs(numerical_prefs)
        logprobs = self.all_logprobs(numerical_prefs)
        return -tf.reduce_sum(probs * logprobs, axis=1)


class GaussianPolicyMX(BaseMixin):
    """Gaussian policy for PG with continuous actions
    """

    def __init__(self):
        BaseMixin.__init__(self)

    def act(self, state, greedy):
        """Epsilon greedy policy:
        """
        numerical_prefs = self.model([state])
        act = self.sample(numerical_prefs, greedy)[0]
        return act

    def sample(self, numerical_prefs, greedy):
        means, logstds = tf.split(numerical_prefs, 2, axis=-1)
        stds = tf.exp(logstds)
        stds = 1e-5 if greedy else stds
        return np.random.normal(means, stds)

    def probs(self, numerical_prefs, actions):
        means, logstds = tf.split(numerical_prefs, 2, axis=-1)
        stds = tf.exp(logstds)
        policy = tf.exp(-tf.square(actions - means) /
                        (2 * tf.square(stds))) / (stds * np.sqrt(2 * np.pi))

        return tf.nn.softmax(numerical_prefs)

    def logprobs(self, numerical_prefs, actions):
        """Get log probabilities of policy.
        """
        means, logstds = tf.split(numerical_prefs, 2, axis=-1)
        stds = tf.exp(logstds)
        logprob = -tf.reduce_sum(logstds, axis=-1) \
                  -0.5*np.log(2*np.pi) \
                  -0.5*tf.reduce_sum(tf.square((actions - means)/stds))
        return logprob

    def policy_entropy(self, numerical_prefs):
        means, logstds = tf.split(numerical_prefs, 2, axis=-1)
        entropy = tf.reduce_sum(logstds + 0.5 * np.log(2.0 * np.pi * np.e),
                                axis=-1)
        return entropy


class DistributionalPolicyMX(EpsilonGreedyPolicyMX):
    """
    """

    def atom_probabilities(self, state_batch):
        """Get atom probabilities.
        """
        return tf.nn.softmax(self.model(state_batch))

    def q_values(self, state_batch):
        """
        """
        probs = self.atom_probabilities(state_batch)
        return np.dot(probs, np.transpose(self.z))

    def act(self, state, greedy):
        """Epsilon greedy policy:
        """
        eps_current = max((self.eps_start - self.ts_eps * self.eps_delta),
                          self.eps_end)
        if greedy or np.random.uniform() > eps_current:
            # q_values = self.model([state])[0]
            q_values = self.network([state])[0]
            return np.argmax(q_values)
        else:
            return np.random.choice(self.env.n_actions)
