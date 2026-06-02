"""
Hint: This implementation is inspired by the following one: https://github.com/ericyangyu/PPO-for-Beginners

The file contains the PPO class to train with.
NOTE: All "ALG STEP"s are following the numbers from the original PPO pseudocode.
                It can be found here: https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg
"""

import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical


class PPO:
    """
    This is the PPO class we will use as our model in main.py
    """

    def __init__(self, policy_class, env, **hyperparameters):
        """
        Initializes the PPO model, including hyperparameters.

        Parameters:
                policy_class - the policy class to use for our actor/critic networks.
                env - the environment to train on.
                hyperparameters - all extra arguments passed into PPO that should be hyperparameters.

        Returns:
                None
        """
        # Make sure the environment is compatible with our code
        assert type(env.observation_space) == gym.spaces.Box
        assert type(env.action_space) == gym.spaces.Discrete

        # Initialize hyperparameters for training with PPO
        self._init_hyperparameters(hyperparameters)

        # Extract environment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = (
            env.action_space.n
        )  # number of discrete actions (3: short/hold/long)

        # Initialize actor and critic networks
        self.actor = policy_class(self.obs_dim, self.act_dim)  # ALG STEP 1
        self.critic = policy_class(self.obs_dim, 1)

        # Initialize optimizers for actor and critic
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # Per-iteration scratch state (cleared at end of every iteration)
        self.logger = {
            "delta_t": time.time_ns(),
            "t_so_far": 0,  # timesteps so far
            "i_so_far": 0,  # iterations so far
            "batch_lens": [],  # episodic lengths in batch
            "batch_rews": [],  # episodic returns in batch
            "actor_losses": [],  # losses of actor network in current iteration
            "critic_losses": [],  # losses of critic network in current iteration
            "entropies": [],  # mean policy entropy per inner update
        }

        # Persistent training history: one entry per PPO iteration.
        # Read from outside after .learn() to plot curves.
        self.history = {
            "iteration": [],
            "t_so_far": [],
            "avg_ep_rew": [],
            "actor_loss": [],
            "critic_loss": [],
            "entropy": [],
        }

    def learn(self, total_timesteps):
        """
        Train the actor and critic networks. Here is where the main PPO algorithm resides.

        Parameters:
                total_timesteps - the total number of timesteps to train for

        Return:
                None
        """
        t_so_far = 0  # Timesteps simulated so far
        i_so_far = 0  # Iterations ran so far
        while t_so_far < total_timesteps:  # ALG STEP 2
            # Collect a fresh batch of rollouts under the current policy
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = (
                self.rollout()
            )  # ALG STEP 3

            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)

            # Increment the number of iterations
            i_so_far += 1

            # Logging timesteps so far and iterations so far
            self.logger["t_so_far"] = t_so_far
            self.logger["i_so_far"] = i_so_far

            # Calculate advantage at k-th iteration
            V, _, _ = self.evaluate(batch_obs, batch_acts)
            A_k = batch_rtgs - V.detach()  # ALG STEP 5

            # One of the only tricks I use that isn't in the pseudocode. Normalizing advantages
            # isn't theoretically necessary, but in practice it decreases the variance of
            # our advantages and makes convergence much more stable and faster. I added this because
            # solving some environments was too unstable without it.
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # This is the loop where we update our network for some n epochs
            for _ in range(self.n_updates_per_iteration):  # ALG STEP 6 & 7
                # Calculate V_phi, pi_theta(a_t | s_t), and the per-sample policy entropy
                V, curr_log_probs, entropy = self.evaluate(batch_obs, batch_acts)

                # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
                # NOTE: we just subtract the logs, which is the same as
                # dividing the values and then canceling the log with e^log.
                # For why we use log probabilities instead of actual probabilities,
                # here's a great explanation:
                # https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
                # TL;DR makes gradient ascent easier behind the scenes.
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate losses.
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                # Actor loss = clipped-surrogate loss  -  entropy_coef * H(pi)
                # Subtracting entropy from the loss = rewarding higher entropy
                # (Adam minimizes loss, so minimizing -H = maximizing H).
                policy_loss = (-torch.min(surr1, surr2)).mean()
                entropy_bonus = entropy.mean()
                actor_loss = policy_loss - self.entropy_coef * entropy_bonus
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                # Calculate gradients and perform backward propagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                # Log actor / critic losses + entropy for this inner update
                self.logger["actor_losses"].append(actor_loss.detach())
                self.logger["critic_losses"].append(critic_loss.detach())
                self.logger["entropies"].append(entropy_bonus.detach())

            # Persist this iteration's metrics into self.history (no printing)
            self._log_summary()

            # Save our model if it's time
            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), "./ppo_actor.pth")
                torch.save(self.critic.state_dict(), "./ppo_critic.pth")

    def rollout(self):
        """
        Collect a fresh batch of rollouts under the current policy.
        Since PPO is on-policy, we re-collect each iteration.

        Return:
                batch_obs       - shape (number of timesteps, observation dim)
                batch_acts      - shape (number of timesteps,)  -- discrete action indices
                batch_log_probs - shape (number of timesteps,)
                batch_rtgs      - shape (number of timesteps,)
                batch_lens      - shape (number of episodes,)
        """
        # Batch data. For more details, check function header.
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

        # Episodic data. Keeps track of rewards per episode, will get cleared
        # upon each new episode
        ep_rews = []

        t = 0  # Keeps track of how many timesteps we've run so far this batch

        # Keep simulating until we've run more than or equal to specified timesteps per batch
        while t < self.timesteps_per_batch:
            ep_rews = []  # rewards collected per episode

            # Reset the environment at the start of each episode
            obs, _ = self.env.reset()
            done = False

            # Run an episode for a maximum of max_timesteps_per_episode timesteps
            for ep_t in range(self.max_timesteps_per_episode):
                # If render is specified, render the environment
                if (
                    self.render
                    and (self.logger["i_so_far"] % self.render_every_i == 0)
                    and len(batch_lens) == 0
                ):
                    self.env.render()

                t += 1  # Increment timesteps ran this batch so far

                # Track observations in this batch
                batch_obs.append(obs)

                # Calculate action and make a step in the env.
                # Note that rew is short for reward.
                action, log_prob = self.get_action(obs)
                obs, rew, terminated, truncated, _ = self.env.step(action)

                # Don't really care about the difference between terminated or truncated in this, so just combine them
                done = terminated | truncated

                # Track recent reward, action, and action log probability
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                # If the environment tells us the episode is terminated, break
                if done:
                    break

            # Track episodic lengths and rewards
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

        # Reshape data as tensors in the shape specified in function description, before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(
            batch_acts, dtype=torch.long
        )  # discrete action indices
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rtgs = self.compute_rtgs(batch_rews)  # ALG STEP 4

        # Log the episodic returns and episodic lengths in this batch.
        self.logger["batch_rews"] = batch_rews
        self.logger["batch_lens"] = batch_lens

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def compute_rtgs(self, batch_rews):
        """
        Compute the Reward-To-Go of each timestep in a batch given the rewards.

        Parameters:
                batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

        Return:
                batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
        """
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []

        # Iterate through each episode
        for ep_rews in reversed(batch_rews):

            discounted_reward = 0  # The discounted reward so far

            # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
            # discounted return (think about why it would be harder starting from the beginning)
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs

    def get_action(self, obs):
        """
        Queries a discrete action from the actor network, should be called from rollout.

        Parameters:
                obs - the observation at the current timestep

        Return:
                action - the discrete action index (int): 0=short, 1=hold, 2=long
                log_prob - the log probability of the selected action under the policy
        """
        # Actor outputs raw logits over the 3 discrete actions
        logits = self.actor(obs)

        # Categorical applies softmax internally to turn logits into action probabilities
        dist = Categorical(logits=logits)

        # Sample an action from the distribution (e.g. probs [0.1, 0.2, 0.7] => action=2 ~70% of the time)
        action = dist.sample()

        # Log-probability of the chosen action under the current policy
        log_prob = dist.log_prob(action)

        # Return as Python int (env expects a scalar for Discrete) and detached log-prob
        return int(action.item()), log_prob.detach()

    def evaluate(self, batch_obs, batch_acts):
        """
        Estimate the values of each observation, and the log probs of
        each action in the most recent batch with the most recent
        iteration of the actor network. Should be called from learn.

        Parameters:
                batch_obs  - tensor of shape (timesteps, observation dim)
                batch_acts - long tensor of shape (timesteps,) with discrete action indices

        Return:
                V         - predicted values of batch_obs, shape (timesteps,)
                log_probs - log probabilities of batch_acts under the current policy
                entropy   - per-sample entropy of the current policy (for the entropy bonus)
        """
        # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        V = self.critic(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most recent actor network.
        # This segment of code is similar to that in get_action()
        logits = self.actor(batch_obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(batch_acts)
        entropy = dist.entropy()

        # Return the value vector V of each observation in the batch,
        # log probabilities log_probs of each action, and per-sample entropy.
        return V, log_probs, entropy

    def _init_hyperparameters(self, hyperparameters):
        """
        Initialize default and custom values for hyperparameters

        Parameters:
                hyperparameters - the extra arguments included when creating the PPO model, should only include
                                                        hyperparameters defined below with custom values.

        Return:
                None
        """
        # Initialize default values for hyperparameters
        # Algorithm hyperparameters
        self.timesteps_per_batch = 4800  # Number of timesteps to run per batch
        self.max_timesteps_per_episode = 1600  # Max number of timesteps per episode
        self.n_updates_per_iteration = (
            5  # Number of times to update actor/critic per iteration
        )
        self.lr = 0.005  # Learning rate of actor optimizer
        self.gamma = (
            0.95  # Discount factor to be applied when calculating Rewards-To-Go
        )
        self.clip = 0.2  # Recommended 0.2, helps define the threshold to clip the ratio during SGA
        self.entropy_coef = (
            0.01  # Coefficient on the entropy bonus in the actor loss (0 disables it)
        )

        # Miscellaneous parameters
        self.render = True  # If we should render during rollout
        self.render_every_i = 10  # Only render every n iterations
        self.save_freq = 10  # How often we save in number of iterations
        self.seed = (
            None  # Sets the seed of our program, used for reproducibility of results
        )

        # Change any default values to custom values for specified hyperparameters
        for param, val in hyperparameters.items():
            exec("self." + param + " = " + str(val))

        # Sets the seed if specified
        if self.seed != None:
            # Check if our seed is valid first
            assert type(self.seed) == int

            # Set the seed
            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")

    def _log_summary(self):
        """
        Persist this iteration's metrics into self.history (for later plotting)
        and reset the per-iteration scratch state. Does not print -- read
        self.history from the outside after training to visualize the run.
        """
        self.logger["delta_t"] = time.time_ns()

        avg_ep_rew = float(np.mean([np.sum(ep) for ep in self.logger["batch_rews"]]))
        avg_actor_l = float(
            np.mean([l.float().mean().item() for l in self.logger["actor_losses"]])
        )
        avg_critic_l = float(
            np.mean([l.float().mean().item() for l in self.logger["critic_losses"]])
        )
        avg_entropy = float(
            np.mean([h.float().mean().item() for h in self.logger["entropies"]])
        )

        self.history["iteration"].append(self.logger["i_so_far"])
        self.history["t_so_far"].append(self.logger["t_so_far"])
        self.history["avg_ep_rew"].append(avg_ep_rew)
        self.history["actor_loss"].append(avg_actor_l)
        self.history["critic_loss"].append(avg_critic_l)
        self.history["entropy"].append(avg_entropy)

        # Reset batch-specific scratch
        self.logger["batch_lens"] = []
        self.logger["batch_rews"] = []
        self.logger["actor_losses"] = []
        self.logger["critic_losses"] = []
        self.logger["entropies"] = []
