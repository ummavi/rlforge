import numpy as np


def uniform_random_choice(choices):
    """Make a random choice from choices.
    """
    random_index = np.random.randint(len(choices))
    return choices[random_index]
