import torch


class AsynchronousRolloutBuffer:
    """
    Asynchronous Replay Buffer for ETD-MAPPO.

    Handles asynchronous credit assignment by accumulating rewards during
    dormant periods and distributing them to the next active decision point.
    Implements Asynchronous Generalized Advantage Estimation (GAE) to
    correctly compute advantages across variable time intervals.
    """

    def __init__(self, num_steps, obs_dim, action_shape, device):
        self.num_steps = num_steps
        self.device = device

        # PPO Memory
        self.obs = torch.zeros((num_steps, obs_dim), dtype=torch.float32).to(device)
        self.actions = torch.zeros((num_steps, *action_shape), dtype=torch.float32).to(device)
        self.logprobs = torch.zeros(num_steps, dtype=torch.float32).to(device)
        self.rewards = torch.zeros(num_steps, dtype=torch.float32).to(device)
        self.values = torch.zeros(num_steps, dtype=torch.float32).to(device)
        self.dones = torch.zeros(num_steps, dtype=torch.float32).to(device)

        # Asynchronous Tracking
        self.active_masks = torch.zeros(num_steps, dtype=torch.float32).to(device)
        self.pending_reward = 0.0

        self.step = 0

    def reset(self):
        """Reset the buffer for a new rollout."""
        self.step = 0
        self.pending_reward = 0.0

    def insert_step(self, obs, action, logprob, value, reward, done, active_mask):
        """
        Insert a single timestep into the buffer.

        Handles reward accumulation for sleeping agents. If an agent is asleep
        (active_mask=0), the reward is stored in pending_reward. When the agent
        wakes up (active_mask=1), it receives the accumulated reward.
        """
        if self.step >= self.num_steps:
            raise IndexError('Buffer is full. Cannot insert more steps.')

        self.obs[self.step] = obs
        self.actions[self.step] = action
        self.logprobs[self.step] = logprob
        self.values[self.step] = value
        self.dones[self.step] = done
        self.active_masks[self.step] = active_mask

        if active_mask == 0.0:
            # Agent is asleep: accumulate reward, log 0 for this step
            self.pending_reward += reward
            self.rewards[self.step] = 0.0
        else:
            # Agent is awake: receive current + accumulated reward
            total_reward = reward + self.pending_reward
            self.rewards[self.step] = total_reward
            self.pending_reward = 0.0

        self.step += 1

    def compute_gae(self, next_value, next_done, gamma, gae_lambda):
        """
        Compute Asynchronous Generalized Advantage Estimation (GAE).

        Unlike standard GAE, this implementation skips over dormant frames,
        calculating advantages based on the next *valid* (awake) value.
        """
        advantages = torch.zeros(self.num_steps, dtype=torch.float32).to(self.device)

        last_valid_gae = 0.0
        next_valid_value = next_value
        next_valid_nonterminal = 1.0 - next_done

        for t in reversed(range(self.num_steps)):
            mask = self.active_masks[t]

            if mask == 0.0:
                advantages[t] = 0.0
                continue

            # Calculate delta using the next valid value (jumping over sleep steps)
            delta = self.rewards[t] + gamma * next_valid_value * next_valid_nonterminal - self.values[t]
            advantages[t] = delta + gamma * gae_lambda * next_valid_nonterminal * last_valid_gae

            # Update pointers for the next backward step
            last_valid_gae = advantages[t]
            next_valid_value = self.values[t]
            next_valid_nonterminal = 1.0 - self.dones[t]

        returns = advantages + self.values
        return advantages, returns
