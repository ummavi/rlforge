import numpy as np
import chainer
from chainer import functions as F


from rlforge.agents.base_agent import BaseAgent
from rlforge.modules.policies import SoftmaxPolicyMX, GaussianPolicyMX
from rlforge.common.utils import discounted_returns


class REINFORCEAgent(BaseAgent):
    """Simple Episodic REINFORCE (Williams92) agent

    The agent works with both discrete and continuous actions
    depending on the policy parameterization.

    Examples of discrete and continuous action agents using a
    Softmax and Gaussian policy parameterization are given below

    See examples/reinforce.py for example agents
    """

    def __init__(self,
                 environment,
                 model,
                 policy_learning_rate,
                 baseline=None,
                 baseline_learning_rate=None,
                 gamma=0.9,
                 entropy_coeff=0.0,
                 optimizer=None,
                 experiment=None):
        """
        Parameters:
        model: A callable model with the final layer being identity.
        baseline ((optional) Value Function Object): Baseline function
        experiment ((optional) Sacred expt): Logs metrics to sacred as well
        optimizer (optional chainer.optimizers.*): Optimizer to use
        """
        BaseAgent.__init__(self, environment, experiment)

        self.model = model
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff

        if optimizer is None:
            self.optimizer = chainer.optimizers.RMSpropGraves(
                policy_learning_rate)
            self.optimizer = self.optimizer.setup(self.model)
        else:
            self.optimizer = optimizer

        self.post_episode_hooks.append(self.learn)

        if baseline is None:
            # If baseline isn't set, use a constant 0 baseline.
            self.baseline = lambda x: np.zeros(len(x))
        else:
            self.baseline = baseline
            self.post_episode_hooks.append(self.learn_baseline)

        self.model_list = [self.model, self.baseline]

    def learn_baseline(self, global_episode_ts, episode_data):
        """ Perform a step of training for the Monte-Carlo Value-function
        estimator.

        post_step_hook Parameters:
            global_step_ts (int)
            episode_data (Episode)
        """
        self.baseline.update_mc([episode_data])
        # self.baseline.update_td([episode_data])

    def learn(self, global_episode_ts, episode_data):
        """Perform one step of learning with a batch of data

        post_step_hook Parameters:
            global_step_ts (int)
            episode_data (Episode)
        """

        returns = discounted_returns(episode_data.rewards, gamma=self.gamma)
        states, actions = episode_data.observations[:-1], episode_data.actions

        returns = returns - self.baseline(states).array

        # Obtain the numerical preferences from the model.
        # .. Depending on the policy, this could be output to
        # .. the softmax or the parameters of the gaussian
        # .. the name's from Sutton & Barto 2nd ed.
        numerical_prefs = self.model(states)
        logprobs = self.logprobs(numerical_prefs, actions)

        average_entropy = F.mean(
            self.policy_entropy(numerical_prefs))

        losses = -F.sum(logprobs * returns)
        losses -= self.entropy_coeff * average_entropy

        self.model.cleargrads()
        losses.backward()
        self.optimizer.update()

        self.logger.log_scalar("episode.average.entropy",
                               float(average_entropy.array))
        self.logger.log_scalar("train.loss", np.mean(losses.array))


class REINFORCEDiscreteAgent(SoftmaxPolicyMX, REINFORCEAgent):
    def __init__(self, **kwargs):
        assert kwargs["environment"].action_space == "discrete"
        REINFORCEAgent.__init__(self, **kwargs)
        SoftmaxPolicyMX.__init__(self)


class REINFORCEContinuousAgent(GaussianPolicyMX, REINFORCEAgent):
    def __init__(self, **kwargs):
        REINFORCEAgent.__init__(self, **kwargs)
        GaussianPolicyMX.__init__(self)
