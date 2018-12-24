import numpy as np
from scipy.signal import savgol_filter


def uniform_random_choice(choices):
    """Make a random choice from choices.
    """
    random_index = np.random.randint(len(choices))
    return choices[random_index]


def smooth_signal(data, window_size=5, poly_order=2):
	"""Smooth 1d signal with a moving average
	"""
	return savgol_filter(data, window_size, poly_order)