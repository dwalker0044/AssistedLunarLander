import math
from typing import Union, Iterable
import numpy as np

from gym.envs.box2d.lunar_lander import LunarLander, SCALE, MAIN_ENGINE_POWER, SIDE_ENGINE_AWAY, VIEWPORT_H, \
    VIEWPORT_W, SIDE_ENGINE_POWER, SIDE_ENGINE_HEIGHT, LEG_DOWN, FPS


class LunarLanderCustom(LunarLander):
    def __init__(self):
        super().__init__()
        self.prev_shaping = None
        self.prev_vy = 0.0

    def step(self, action: Union[np.ndarray, Iterable, int, float]):
        if self.continuous:
            action = np.clip(action, -1, +1).astype(np.float32)
        else:
            assert self.action_space.contains(action), "%r (%s) invalid " % (action, type(action))

        # Engines
        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0])
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        m_power = 0.0
        if (self.continuous and action[0] > 0.0) or (not self.continuous and action == 2):
            # Main engine
            if self.continuous:
                m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5  # 0.5..1.0
                assert 0.5 <= m_power <= 1.0
            else:
                m_power = 1.0

            # 4 is move a bit downwards, +-2 for randomness
            ox = tip[0] * (4 / SCALE + 2 * dispersion[0]) + side[0] * dispersion[1]
            oy = -tip[1] * (4 / SCALE + 2 * dispersion[0]) - side[1] * dispersion[1]
            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)

            # particles are just a decoration, 3.5 is here to make particle speed adequate
            p = self._create_particle(3.5, impulse_pos[0], impulse_pos[1],
                                      m_power)
            p.ApplyLinearImpulse((ox * MAIN_ENGINE_POWER * m_power, oy * MAIN_ENGINE_POWER * m_power), impulse_pos,
                                 True)
            self.lander.ApplyLinearImpulse((-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power),
                                           impulse_pos, True)

        s_power = 0.0
        if (self.continuous and np.abs(action[1]) > 0.5) or (not self.continuous and action in [1, 3]):
            # Orientation engines
            if self.continuous:
                direction = np.sign(action[1])
                s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
                assert 0.5 <= s_power <= 1.0
            else:
                direction = action - 2
                s_power = 1.0
            ox = tip[0] * dispersion[0] + side[0] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE)
            oy = -tip[1] * dispersion[0] - side[1] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE)
            impulse_pos = (self.lander.position[0] + ox - tip[0] * 17 / SCALE,
                           self.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE)
            p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
            p.ApplyLinearImpulse((ox * SIDE_ENGINE_POWER * s_power, oy * SIDE_ENGINE_POWER * s_power), impulse_pos,
                                 True)
            self.lander.ApplyLinearImpulse((-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power),
                                           impulse_pos, True)

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        pos = self.lander.position
        vel = self.lander.linearVelocity
        state = [
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
        ]
        self.prev_vy = vel.y
        assert len(state) == 8

        done, reward, descent_reward = self.calculate_reward(state, m_power, s_power)
        return np.array(state, dtype=np.float32), (reward, descent_reward), done, {}

    def calculate_reward(self, state, m_power, s_power):
        reward = 0
        shaping = \
            - 100 * np.sqrt(state[0] * state[0] + state[1] * state[1]) \
            - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3]) \
            - 100 * abs(state[4]) + 10 * state[6] + 10 * state[7]
        # (Above) And ten points for legs contact, the idea is if you
        # lose contact again after landing, you get negative reward

        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        # less fuel spent is better, about -30 for heuristic landing
        reward -= m_power * 0.30
        reward -= s_power * 0.03

        # Velocity should reduce as you approach the surface.
        # If you are far away, maximum reward is achieved simply by going down.
        a = (state[3] - self.prev_vy) * (VIEWPORT_H / SCALE / 2) / FPS

        descent_reward = 0

        if state[1] < 1.3:
            descent_reward += (1 - np.abs(state[0])) * 2

            # If you de-accelerate, get reward.
            # if a < 0:
            #     descent_reward += 1
            if state[1] < 1 and state[3] < 0.4:
                descent_reward += (1 - np.abs(state[3]))

            # The nearer you get to surface, get more reward.
            # descent_reward += (1 - state[1]) * 0.5

            # If moving slowly, get reward.
            # if state[3] > -0.5 and state[3] < 0:
            #     descent_reward += 1
            #
            # if state[3] > -0.3 and state[3] < 0:
            #     descent_reward += 2
            #
            # if state[3] > -0.1 and state[3] < 0:
            #     descent_reward += 4

        done = False
        if self.game_over or abs(state[0]) >= 1.0:
            done = True
            reward = -100
            descent_reward = 0
        if not self.lander.awake:
            done = True
            reward = +100
            descent_reward = 0
        if self.legs[0].ground_contact or self.legs[1].ground_contact:
            descent_reward = 0

        return done, reward, descent_reward
