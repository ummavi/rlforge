import chainer
import numpy as np

from chainer import functions as F

from rlforge.agents.base_agent import BaseAgent
from rlforge.modules.policies import SoftmaxPolicyMX, GaussianPolicyMX


class A2CAgent(BaseAgent):
    """
    A2C paper implementation as described in
    (https://blog.openai.com/baselines-acktr-a2c/
    https://arxiv.org/abs/1602.01783)

    Note: Instead of doing parallel rollouts, gradients are batched
    and updated every `n_workers` episodes.

    See examples/a2c.py for example agent

    TODO:
    * Parallel runners
    * GAE (https://arxiv.org/abs/1506.02438)
    """

    def __init__(self,
                 environment,
                 model,
                 model_learning_rate,
                 v_function_coeff,
                 gamma=0.9,
                 entropy_coeff=3,
                 n_workers=2,
                 optimizer=None,
                 experiment=None):
        """
        Parameters:
        model: A callable model with the final layer being identity.
        v_function: A callable model of the state-value function
        v_function_coeff: Coeff. the V-function loss should be scaled by
        experiment [(optional) Sacred expt]: Logs metrics to sacred as well
        """
        BaseAgent.__init__(self, environment, experiment)

        self.gamma = gamma
        self.model = model
        self.n_workers = n_workers
        self.entropy_coeff = entropy_coeff

        if optimizer is None:
            self.optimizer = chainer.optimizers.RMSpropGraves(
                    model_learning_rate)
            self.optimizer = self.optimizer.setup(self.model)
            self.model.cleargrads()  # Clear gradients for first round

        else:
            self.optimizer = optimizer

        self.post_episode_hooks.append(self.learn)

        self.v_function_coeff = v_function_coeff

        self.model_list = [self.model]

    def learn(self, global_episode_ts, episode_data):
        """Perform one step of learning with a batch of data

        post_step_hook Parameters:
            global_step_ts (int)
            episode_data (Episode)
        """

        states, actions = episode_data.observations, episode_data.actions
        q_sts = self.model.nstep_bootstrapped_value(episode_data)
        # Terminal state's value is 0.
        q_sts = np.squeeze(np.float32(np.concatenate((q_sts, [0.0]))))

        # numerical_prefs, v_sts = self.model(states)
        numerical_prefs = self.model.policy(states)
        with chainer.no_backprop_mode():
            v_sts = self.model.v(states)
            v_sts = F.squeeze(v_sts).array

        numerical_prefs = numerical_prefs[:-1]  # Ignore A_{T}

        advantage = q_sts - v_sts

        # Policy Loss
        logprobs = self.logprobs(numerical_prefs, actions)
        average_entropy = F.mean(
            self.policy_entropy(numerical_prefs))

        loss_policy = -F.mean(logprobs * advantage[:-1])
        loss_policy -= self.entropy_coeff * average_entropy

        # Value loss
        loss_value = F.mean_squared_error(q_sts, v_sts)
        losses = loss_policy + self.v_function_coeff * loss_value

        # Create backward and accumulate.
        losses.backward()
        if (global_episode_ts) % self.n_workers == 0:
            # Apply and reset gradients every n_worker steps
            self.optimizer.update()
            self.model.cleargrads()

        self.logger.log_scalar("episode_average_entropy",
                               float(average_entropy.array))
        self.logger.log_scalar("episode_average_advantage",
                               np.mean(advantage))
        self.logger.log_scalar("episode_losses", float(losses.array))


class A2CDiscreteAgent(SoftmaxPolicyMX, A2CAgent):
    def __init__(self, **kwargs):
        assert kwargs["environment"].action_space == "discrete"
        A2CAgent.__init__(self, **kwargs)
        SoftmaxPolicyMX.__init__(self)


class A2CContinuousAgent(GaussianPolicyMX, A2CAgent):
    def __init__(self, **kwargs):
        A2CAgent.__init__(self, **kwargs)
        GaussianPolicyMX.__init__(self)
