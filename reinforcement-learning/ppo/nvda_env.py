"""
Hint: This implementation is inspired by the following one: https://github.com/ericyangyu/PPO-for-Beginners

NVDA daily-trading Gym environment for PPO.

Reward (per task spec):
    r_t^net = a_{t-1} * return_t  -  cost * |a_t - a_{t-1}|

Inputs:
    features: precomputed per-day state vector (e.g. PCA / AE codes).
    returns:  per-day NVDA simple daily return (pct_change).
    action:   Discrete(3). 0 = short (-1), 1 = hold (0), 2 = long (+1).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class NVDATradingEnv(gym.Env):
    metadata = {"render_modes": []}

    POS_MAP = np.array([-1.0, 0.0, +1.0], dtype=np.float32)

    def __init__(
        self,
        features: np.ndarray,
        returns: np.ndarray,
        cost: float = 0.001,
        max_steps: int = 252,
        random_start: bool = True,
    ):
        super().__init__()
        assert len(features) == len(returns), "features and returns must align by index"
        assert len(returns) > max_steps + 1, "need at least max_steps+1 rows of data"

        self.features = np.asarray(features, dtype=np.float32)
        self.returns = np.asarray(returns, dtype=np.float32)
        self.cost = float(cost)
        self.max_steps = int(max_steps)
        self.random_start = bool(random_start)

        n_feat = self.features.shape[1]
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_feat + 1,), dtype=np.float32
        )

        self.t0 = 0
        self.t = 0
        self.prev_action = 0.0

    def _obs(self) -> np.ndarray:
        feat = self.features[self.t0 + self.t]
        return np.concatenate([feat, [self.prev_action]]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        max_t0 = len(self.returns) - self.max_steps - 1
        if self.random_start and max_t0 > 0:
            self.t0 = int(self.np_random.integers(0, max_t0))
        else:
            self.t0 = 0
        self.t = 0
        self.prev_action = 0.0
        return self._obs(), {}

    def step(self, action):
        a_t = float(self.POS_MAP[int(action)])
        ret_next = float(self.returns[self.t0 + self.t + 1])

        reward = self.prev_action * ret_next - self.cost * abs(a_t - self.prev_action)

        self.prev_action = a_t
        self.t += 1
        terminated = False
        truncated = self.t >= self.max_steps
        return self._obs(), float(reward), terminated, truncated, {}

    def render(self):
        pass
