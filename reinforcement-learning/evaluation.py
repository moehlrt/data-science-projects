"""
Run a trading policy through the environment for evaluation.

Steps the policy through the env day by day and returns the per-day
positions (-1/0/+1) and rewards, which the notebook turns into the
cumulative PnL curve and the performance metrics.
"""

import numpy as np


def evaluate(env, action_fn):
    """Run action_fn through env and return (positions, rewards) as arrays."""
    obs, _ = env.reset()
    positions, rewards, done = [], [], False
    while not done:
        action = action_fn(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        positions.append(action - 1)
        rewards.append(reward)
        done = terminated or truncated
    return np.array(positions), np.array(rewards)
