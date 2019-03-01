import numpy as np
import tensorflow as tf
import multiprocessing as mp
tf.enable_eager_execution()

from sacred import Experiment
from sacred.observers import MongoObserver
from hyperopt import fmin, tpe, hp

from rlforge.environments.environment import GymEnv
from rlforge.agents.a2c import A2CAgent, A2CContinuousAgent
from rlforge.common.value_functions import ValuePolicyNetworkDense

# Define a config space
config_space = {
    "env_name": "MountainCarContinuous-v0",
    "n_train_episodes": 150,
    "seeds": [1, 2, 3, 4, 5],

    "model_layer_sizes": hp.choice('model_layer_sizes',
                                   [[128, 128], [256, 256]]),
    "n_steps": hp.choice('n_steps', [2, 4, 6, 8]),
    "n_workers": hp.choice('n_workers', [2, 4, 6, 8]),
    "gamma": hp.choice('gamma', [0.7, 0.8, 0.9, 0.95, 0.99]),
    "activation_fn": hp.choice('activation_fn', ["tanh", "relu"]),
    "entropy_coeff": hp.choice("entropy_coeff", [0.001, 0.01, 0.1, 1., 2, 3]),
    "v_function_coeff": hp.choice('v_function_coeff',
                                  [0.01, 0.1, 1.0, 2.0, 3.0]),
    "model_learning_rate": hp.choice("model_learning_rate",
                                     [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]),
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
    def agent_setup(env_name, seed, gamma, model_layer_sizes,
                    model_learning_rate, v_function_coeff, entropy_coeff,
                    n_workers, n_train_episodes, activation_fn, n_steps):

        env = GymEnv(env_name)

        np.random.seed(seed)
        tf.set_random_seed(seed)
        env.env.seed(seed)

        network_config = dict(layer_sizes=model_layer_sizes,
                              activation=activation_fn)
        output_sizes = [2 * env.n_actions, 1]
        combined_model = ValuePolicyNetworkDense(network_config,
                                                 output_sizes, gamma,
                                                 n_steps=n_steps)

        agent = A2CContinuousAgent(env, combined_model, model_learning_rate,
                                   v_function_coeff=v_function_coeff,
                                   gamma=gamma,
                                   entropy_coeff=entropy_coeff,
                                   n_workers=n_workers,
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
        fn=main,
        space=config_space,
        algo=tpe.suggest,
        max_evals=100)
    print("Best configuration:", best)
    """
    Best configuration: {'activation_fn': 0, 'entropy_coeff': 1, 'gamma': 4, 'model_layer_sizes': 1, 'model_learning_rate': 0, 'n_steps': 1, 'n_workers': 0, 'v_function_coeff': 0}
    best_config = {
        "env_name": "MountainCarContinuous-v0",
        "n_train_episodes": 150,
        "seeds": [1, 2, 3, 4, 5],
        "model_layer_sizes": [256, 256],
        "n_steps": 4,
        "n_workers": 2,
        "gamma": 0.99,
        "activation_fn": "tanh",
        "entropy_coeff": 0.01,
        "v_function_coeff": 0.01,
        "model_learning_rate": 1e-3
    }
    """
