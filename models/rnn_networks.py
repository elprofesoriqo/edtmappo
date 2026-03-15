import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Orthogonal initialization for neural network layers.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
    """
    ETD-MAPPO Actor Network.

    Responsible for generating action logits and computing policy entropy
    used for the uncertainty-driven frame-skipping mechanism.
    """

    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )

    def forward(self, x):
        """Forward pass through the actor network."""
        return self.network(x)

    def get_action_and_sleep_ticks(self, x, entropy_threshold, max_sleep_ticks=3, action=None):
        """
        Sample action and determine sleep duration based on entropy.

        Args:
            x: Observation tensor.
            entropy_threshold: Threshold for determining uncertainty.
            max_sleep_ticks: Maximum frames to skip if confident.
            action: Optional action to evaluate.

        Returns:
            Tuple of (action, log_prob, entropy, sleep_ticks).
        """
        logits = self(x)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        # Determine sleep ticks based on uncertainty
        if isinstance(max_sleep_ticks, int):
            max_sleep_ticks_tensor = torch.full_like(entropy, max_sleep_ticks, dtype=torch.int32, device=x.device)
        else:
            max_sleep_ticks_tensor = max_sleep_ticks.to(x.device)

        sleep_ticks = torch.where(
            entropy <= entropy_threshold, max_sleep_ticks_tensor, torch.zeros_like(max_sleep_ticks_tensor)
        )

        return action, log_prob, entropy, sleep_ticks


class Critic(nn.Module):
    """
    Centralized Critic Network.

    Evaluates the global state value V(s) for the multi-agent system.
    """

    def __init__(self, global_obs_dim):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(global_obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def forward(self, global_obs):
        """Forward pass through the critic network."""
        return self.network(global_obs)


class AsynchronousGRUActor(nn.Module):
    """
    Recurrent Actor Network with Asynchronous Execution support.

    Incorporates a GRU cell that selectively updates its hidden state
    only for active agents, preserving state consistency during sleep.
    """

    def __init__(self, obs_dim, action_dim, hidden_size=64):
        super().__init__()
        self.fc1 = layer_init(nn.Linear(obs_dim, hidden_size))

        self.gru = nn.GRUCell(hidden_size, hidden_size)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1.0)

        self.fc2 = layer_init(nn.Linear(hidden_size, action_dim), std=0.01)

    def forward(self, x, hidden_states, active_masks):
        """
        Forward pass with selective GRU updates.

        Args:
            x: Observation tensor.
            hidden_states: Previous hidden states.
            active_masks: Mask indicating active (1) or sleeping (0) agents.
        """
        x = torch.tanh(self.fc1(x))

        masks_bool = active_masks.squeeze().bool()
        new_hidden = hidden_states.clone()

        # Update hidden state only for active agents
        if masks_bool.dim() == 0:
            if masks_bool.item():
                new_hidden = self.gru(x, hidden_states)
        elif masks_bool.any():
            new_hidden[masks_bool] = self.gru(x[masks_bool], hidden_states[masks_bool])

        logits = self.fc2(new_hidden)
        return logits, new_hidden

    def get_action_and_sleep_ticks(
        self, x, hidden_states, active_masks, entropy_threshold, max_sleep_ticks=3, action=None
    ):
        """
        Sample action and determine sleep duration with recurrent state.
        """
        logits, new_hidden = self.forward(x, hidden_states, active_masks)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        if isinstance(max_sleep_ticks, int):
            max_sleep_ticks_tensor = torch.full_like(entropy, max_sleep_ticks, dtype=torch.int32, device=x.device)
        else:
            max_sleep_ticks_tensor = max_sleep_ticks.to(x.device)

        sleep_ticks = torch.where(
            entropy <= entropy_threshold, max_sleep_ticks_tensor, torch.zeros_like(max_sleep_ticks_tensor)
        )

        return action, log_prob, entropy, sleep_ticks, new_hidden


class AsynchronousGRUCritic(nn.Module):
    """
    Recurrent Centralized Critic Network.

    Evaluates global state with support for asynchronous hidden state updates.
    """

    def __init__(self, global_obs_dim, hidden_size=64):
        super().__init__()
        self.fc1 = layer_init(nn.Linear(global_obs_dim, hidden_size))

        self.gru = nn.GRUCell(hidden_size, hidden_size)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1.0)

        self.fc2 = layer_init(nn.Linear(hidden_size, 1), std=1.0)

    def forward(self, global_obs, hidden_states, active_masks):
        """
        Forward pass with selective GRU updates.
        """
        x = torch.tanh(self.fc1(global_obs))
        masks_bool = active_masks.squeeze().bool()

        new_hidden = hidden_states.clone()

        if masks_bool.dim() == 0:
            if masks_bool.item():
                new_hidden = self.gru(x, hidden_states)
        elif masks_bool.any():
            new_hidden[masks_bool] = self.gru(x[masks_bool], hidden_states[masks_bool])

        value = self.fc2(new_hidden)
        return value, new_hidden
