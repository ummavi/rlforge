import numpy as np 


from rlforge.agents.base_agent import BaseMixin
from rlforge.agents.base_agent import BaseAgent

class EpsilonGreedyPolicyMX:
    """Epsilon-Greedy policy with or w/o an epslion schedule.
    """
    def __init__(self,eps_fixed=0.2,episodic_update=False,
                eps_start=None,eps_end=None,ts_eps_end=None,eps_schedule=None):
        """
        Parameters:
        eps_schedule {None | "linear"}: Chooses between a fixed learning rate and a decay schedule
        """
        assert eps_schedule in [None, "linear"]

        self.eps_schedule = eps_schedule

        if eps_schedule is None:
            self.eps_start = self.eps_end = eps_fixed
            self.eps_delta = 0.0

        self.eps_ts = 0 #Note, this ts might be different from the agent's.

        if eps_schedule == "linear":
            assert eps_start is not None
            assert eps_end is not None 
            assert ts_eps_end is not None

            self.eps_delta = float(eps_end-eps_start)/ts_eps_end
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
        eps_current = max((self.eps_start - self.eps_ts*self.eps_delta), self.eps_end)
        if greedy or np.random.uniform()>eps_current:
            q_values = self.model(state)[0]
            return np.argmax(q_values)
        else:
            return np.random.choice(self.env.n_actions)


    def update_eps(self,global_step_ts, step_data):
        """Updates the eps-greedy timestep. 
        Making this explicit allows us to use the decay for episodes or timesteps.
        If eps_ts==None, it's assumed only one epsilon "step" has occurred.
        """    
        self.eps_ts += 1