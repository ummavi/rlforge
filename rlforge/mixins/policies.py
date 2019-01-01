import numpy as np 
import tensorflow as tf

from rlforge.agents.base_agent import BaseMixin

class EpsilonGreedyPolicyMX(BaseMixin):
    """Epsilon-Greedy policy with or w/o an epslion schedule
    """
    def __init__(self,eps_fixed=0.2,episodic_update=False,
                eps_start=None,eps_end=None,ts_eps_end=None,eps_schedule=None):
        """
        Parameters:
        eps_schedule {None | "linear"}: Chooses between a fixed learning rate and a decay schedule
        """
        assert eps_schedule in [None, "linear"]
        BaseMixin.__init__(self)


        self.eps_schedule = eps_schedule

        if eps_schedule is None:
            self.eps_start = self.eps_end = eps_fixed
            self.eps_delta = 0.0

        self.ts_eps = 0 #Note, this ts might be different from the agent's.

        if eps_schedule == "linear":
            assert eps_start is not None
            assert eps_end is not None 
            assert ts_eps_end is not None

            self.eps_delta = float(eps_start - eps_end)/ts_eps_end
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
        eps_current = max((self.eps_start - self.ts_eps*self.eps_delta), self.eps_end)
        if greedy or np.random.uniform()>eps_current:
            q_values = self.model([state])[0]
            return np.argmax(q_values)
        else:
            return np.random.choice(self.env.n_actions)


    def update_eps(self,global_any_ts, any_data):
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
        policy = np.float32(tf.nn.softmax(self.model([state]))[0])
        if greedy:
            return np.argmax(policy)
        else:
            return np.random.choice(self.env.n_actions, p=policy)
