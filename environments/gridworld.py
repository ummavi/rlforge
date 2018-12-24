import numpy as np
import matplotlib.pyplot as plt


from .utils import uniform_random_choice


class Gridworld:
    """Class for generic gridworld.
    """
    grid, cur_source_loc, cur_target_loc = None, None, None
    n_steps, cur_loc = None, None

    def __init__(self, gridsize=(10, 10), source_locs=[(0, 0)], target_locs=[(9, 9)],
                 blocked_locs=[(5, 5), (5, 6), (6, 6), (7, 6)], max_steps=None,
                 dense_rewards=False, eps=0.0):
        """
        Parameters:
        max_steps {None or int}: Specify max_steps an episode can have
        eps {float [0,1]}: Specify % of the time the env ignores input
        dense_rewards {bool}: 0/1 reward if False else -1/0 
        """

        self.n_actions = 4
        self.d_observations = [2]

        self.eps = eps
        self.gridsize = gridsize
        self.max_steps = max_steps
        self.source_locs = source_locs
        self.target_locs = target_locs
        self.blocked_locs = blocked_locs
        self.dense_rewards = dense_rewards

    def render(self):
        """Plot a matplotlib-powered visualization
        """
        gs = np.array(self.grid)  # np.array(...) forces it to copy
        gs[self.cur_loc[0], self.cur_loc[1]] = 5
        plt.imshow(gs)
        plt.gca().grid(False)
        plt.show()

    def _get_obs(self):
        """Returns the observation or the current state of the agent
        """
        return self.cur_loc

    def reset(self):
        """Reset the game/start a new game and return the default state
        """
        #Select one of the start states and the goal states from potentially many
        self.cur_source_loc = uniform_random_choice(self.source_locs)
        self.cur_target_loc = uniform_random_choice(self.target_locs)

        self.grid = np.zeros(self.gridsize)  # Zero is free square

        self.grid[self.cur_source_loc[0], self.cur_source_loc[1]] = 1
        for b in self.blocked_locs:
            self.grid[b[0], b[1]] = -1  # -1 = Blocked Square

        # 1 is a goal square
        self.grid[self.cur_target_loc[0], self.cur_target_loc[1]] = 2
        self.cur_loc = self.cur_source_loc
        self.n_steps = 0

        return self.cur_loc

    def step(self, a):
        '''
        Perform an action and take one "step" in the game
        Parameters:
            a: action to take. Must be [0,1,2,3] = [up,right,down,left]

        Return Values:
            obs_n: observation of the next step (s')
            r: Reward for current timestep
            episode_end: Has the current episode ended or not.
        '''
        if not 0 <= a <= 3:
            raise Exception("Invalid action")

        if np.random.uniform()<self.eps:
            a = np.random.choice(self.n_actions)

        self.n_steps += 1

        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        new_loc = (self.cur_loc[0]+moves[a][0], self.cur_loc[1]+moves[a][1])

        # Check if the new location is still within the maze and not blocked
        #.. if it's invalid, stay at the old location
        if new_loc not in self.blocked_locs and  \
                0 <= new_loc[0] < self.gridsize[0] and \
                0 <= new_loc[1] < self.gridsize[1]:
            self.cur_loc = new_loc
        else:
            self.cur_loc = self.cur_loc

        done, reward = False, 0

        if self.cur_loc == self.cur_target_loc:
            done, reward = True, 1

        if self.n_steps == self.max_steps:
            done = True

        if self.dense_rewards:
            reward = -1 if reward==0 else 0

        return np.array(self.cur_loc), reward, done
