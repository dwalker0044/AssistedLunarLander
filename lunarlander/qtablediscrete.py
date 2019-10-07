import numpy as np

EPSILON = 0.5
ALPHA = 0.1
GAMMA = 0.6


class QTable1dDiscrete:
    def __init__(self, x_state, x_bins, n_action_space):
        self.x_state = x_state
        self.x_bins = x_bins

        shape = (self.x_bins.shape[0], n_action_space)
        self.q_table = np.zeros(shape)

    def lookup_state(self, state):
        # Discretize the continuous position of the lander.
        state_value = state[self.x_state]
        state_index = np.digitize(state_value, self.x_bins)

        assert 0 <= state_index < self.x_bins.shape[0], \
            f"OoB array index {state_index}. State value was {state_value}"
        return self.q_table[state_index]

    def update(self, state, next_state, action, reward):
        state_action_space = self.lookup_state(state)
        old_value = state_action_space[action]

        next_max = np.max(self.lookup_state(next_state))

        new_value = (1 - ALPHA) * old_value + ALPHA * (reward + GAMMA * next_max)
        state_action_space[action] = new_value
