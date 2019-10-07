from typing import Union, Iterable
import numpy as np

from gym.envs.box2d.lunar_lander import LunarLander


class LunarLanderCustom(LunarLander):
    def __init__(self):
        super().__init__()

    def step(self, action: Union[np.ndarray, Iterable, int, float]):
        next_state, reward, done, empty_dict = super().step(action)
        upright_reward = self.calculate_additional_rewards(next_state)
        return next_state, (reward, upright_reward), done, empty_dict

    def calculate_additional_rewards(self, state):
        upright_reward = 0

        # Checking that the lander is below a certain point encourages it to fall - otherwise it tends to ascend.
        if state[1] < 1.3:
            theta = np.abs(state[4])
            if theta > 20 * np.pi / 180:
                upright_reward = -theta

        # This helps ensure the lander doe not get reward when the game is over.
        if self.game_over or abs(state[0]) >= 1.0:
            upright_reward = 0
        if not self.lander.awake:
            upright_reward = 0
        if self.legs[0].ground_contact or self.legs[1].ground_contact:
            upright_reward = 0

        return upright_reward
