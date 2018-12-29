import numpy as np



class BaseMixin():
    """Base Mixin template which defines all the hooks. 
    """
    pre_episode_hooks = []
    post_episode_hooks = []

    pre_step_hooks = []
    post_step_hooks = [] 

