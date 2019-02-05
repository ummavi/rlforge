import numpy as np
import tensorflow as tf

import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm

sns.set_style("dark")
sns.set_palette(sns.color_palette("Set1"))


def train_sequential(agent, env, n_episodes, n_restarts=1, seed=1, last_n=5):
    """
    Train an agent sequentially n_restarts times

    parameters:
        n_episodes: number of episodes (per restart) to train
        n_restarts: number of restarts to reset the agent and the environment
        seeds (int or list):

    For multiple restarts, either pass a list of seeds
                        (or)
    Set n_restarts>1. The seeds used are sequential starting from
    the value set for seed
    """

    if type(seed) is int:
        seeds = np.arange(seed, seed+n_restarts)
    else:
        seeds = seed

    all_returns = []
    for seed in seeds:
        np.random.seed(seed)
        tf.set_random_seed(seed)
        env.env.seed(seed)

        agent.reset()

        pbar = tqdm(range(n_episodes))
        for ep in pbar:
            agent.interact(1)
            if ep % last_n == 0:
                n = agent.stats.get_values("episode_returns")[-last_n:]
                pbar.set_description("Latest return: " + str(np.mean(n)))
        all_returns.append(agent.stats.get_values("episode_returns"))

    mean_per_restart = np.mean(all_returns, axis=1)
    mean, std = np.mean(mean_per_restart), np.std(mean_per_restart)
    print("Mean return: {0} ± {1}".format(mean, std))

    all_returns = pd.DataFrame(all_returns).melt()
    sns.lineplot(x="variable", y="value", data=all_returns,
                 estimator=np.mean, ci=95)
    plt.xlabel("Episode")
    plt.ylabel("Average Return")
    plt.tight_layout()
    plt.show()

