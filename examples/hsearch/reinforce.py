import numpy as np
import tensorflow as tf
import multiprocessing as mp
tf.enable_eager_execution()

from sacred import Experiment
from sacred.observers import MongoObserver

from hyperopt import fmin, tpe, hp

from rlforge.agents.pg import REINFORCEAgent, REINFORCEContinuousAgent
from rlforge.common.policy_functions import PolicyNetworkDense
from rlforge.common.value_functions import VNetworkDense
from rlforge.environments.environment import GymEnv

# Define a config space
config_space = {
    "env_name":
    "MountainCarContinuous-v0",
    "n_train_episodes":
    150,
    "gamma":
    hp.choice('gamma', [0.7, 0.8, 0.9, 0.95, 0.99]),
    "policy_layer_sizes": [64, 64],
    "policy_learning_rate":
    hp.choice("policy_learning_rate", [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]),
    "baseline_layer_sizes": [64, 64],
    "entropy_coeff":
    hp.choice("entropy_coeff", [0.001, 0.01, 0.1, 1., 2, 3]),
    "baseline_learning_rate":
    hp.choice("baseline_learning_rate", [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]),
    "seeds": [1, 2, 3, 4, 5],
    "activation_fn":
    hp.choice('activation_fn', ["tanh", "relu"])
}


def run(expt_config, result_q):
    """Create and run an agent with the specified configuration file 
    .. and return the result in the result_q 
    """
    # Set up sacred experiment and initialize a mongo observer.
    ex = Experiment()
    ex.add_config(expt_config)
    ex.observers.append(MongoObserver.create())

    @ex.capture
    def agent_setup(env_name, seed, gamma, policy_layer_sizes,
                    baseline_layer_sizes, policy_learning_rate, entropy_coeff,
                    baseline_learning_rate, n_train_episodes, activation_fn):

        env = GymEnv(env_name)

        np.random.seed(seed)
        tf.set_random_seed(seed)
        env.env.seed(seed)

        policy_config = dict(
            layer_sizes=policy_layer_sizes, activation=activation_fn)
        policy = PolicyNetworkDense(2 * env.n_actions, policy_config)

        value_config = dict(
            layer_sizes=baseline_layer_sizes, activation=activation_fn)
        value_opt = tf.train.AdamOptimizer(baseline_learning_rate)
        value_baseline = VNetworkDense(value_config, gamma, value_opt)

        agent = REINFORCEContinuousAgent(
            env,
            policy,
            policy_learning_rate=policy_learning_rate,
            baseline=value_baseline,
            gamma=gamma,
            entropy_coeff=entropy_coeff,
            experiment=ex)
        return agent, env

    @ex.main
    def agent_train_and_eval(n_train_episodes):
        agent, env = agent_setup()
        _ = agent.interact(n_train_episodes, show_progress=False)
        returns = agent.logger.get_values("episode.returns")
        result_q.put(returns)

    return ex.run()


def main(config):
    """Runs the agent(s) parallelly for each seed and combine the results
    to get average return for the supplied configuration.
    """
    all_procs = []
    result_q = mp.Queue()
    for seed in config["seeds"]:
        config["seed"] = seed
        p = mp.Process(target=run, args=(config, result_q))
        p.start()
        all_procs.append(p)

    for p in all_procs:
        p.join()

    all_returns = [result_q.get() for p in all_procs]
    mean_per_restart = np.mean(all_returns, axis=1)
    mean, std = np.mean(mean_per_restart), np.std(mean_per_restart)
    # Return the negative since we're minimizing the function
    # .. the metric minimized is suggested from Duan et al. (2016)
    return -(mean - std)


if __name__ == "__main__":
    best = fmin(
        fn=main,  # "Loss" function to minimize
        space=config_space,  # Hyperparameter space
        algo=tpe.suggest,  # Tree-structured Parzen Estimator (TPE)
        max_evals=25)
    print("Best configuration:", best)
    """
    Best configuration: {'activation_fn': 0, 'baseline_learning_rate': 1,
                        'entropy_coeff': 0, 'gamma': 0,
                        'policy_learning_rate': 1}
    config_space = {
        "env_name": "MountainCarContinuous-v0",
        "n_train_episodes": 150,
        "gamma": 0.7,
        "policy_layer_sizes": [64, 64],
        "policy_learning_rate": 5e-4,
        "baseline_layer_sizes": [64, 64],
        "entropy_coeff": 0.001,
        "baseline_learning_rate": 5e-4,
        "seeds": [1, 2, 3, 4, 5],
        "activation_fn": "tanh")
    }
    """
