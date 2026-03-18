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

        self.step = 0

    def reset(self):
        """Reset the buffer for a new rollout."""
        self.step = 0

    def insert_step(self, obs, action, logprob, value, reward, done, active_mask):
        """
        Insert a single timestep into the buffer.
        Rewards are stored raw; SMDP temporal discounting is handled during GAE computation.
        """
        if self.step >= self.num_steps:
            raise IndexError('Buffer is full. Cannot insert more steps.')

        self.obs[self.step] = obs
        self.actions[self.step] = action
        self.logprobs[self.step] = logprob
        self.values[self.step] = value
        self.dones[self.step] = done
        self.active_masks[self.step] = active_mask

        # SMDP-aligned execution: Reward is stored precisely where it occurred
        if not isinstance(reward, torch.Tensor):
            reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        self.rewards[self.step] = reward

        self.step += 1

    def compute_gae(self, next_value, next_done, gamma, gae_lambda):
        """
        Compute SMDP-Aligned Generalized Advantage Estimation (GAE).
        
        Properly discounts delayed rewards during sleep and bridges
        the temporal difference equation across N-step sleep gaps using gamma^N.
        """
        advantages = torch.zeros(self.num_steps, dtype=torch.float32).to(self.device)

        last_valid_gae = 0.0
        next_valid_value = next_value
        next_valid_nonterminal = 1.0 - next_done

        # SMDP Gap tracker (how many steps ahead the 'next valid' state is)
        current_gap = 1

        for t in reversed(range(self.num_steps)):
            mask = self.active_masks[t]

            if mask == 0.0:
                advantages[t] = 0.0
                current_gap += 1
                continue

            # Compute SMDP temporally discounted effective reward for the sleep gap
            r_effective = 0.0
            for k in range(current_gap):
                if t + k < self.num_steps:
                    r_effective += (gamma ** k) * self.rewards[t + k]

            # Calculate SMDP-aligned delta bridging the gap
            gamma_gap = gamma ** current_gap
            delta = r_effective + gamma_gap * next_valid_value * next_valid_nonterminal - self.values[t]

            # Formulate Advantage
            # GAE bridges across the gap: (gamma * gae_lambda) ** current_gap
            gae_decay = (gamma * gae_lambda) ** current_gap
            advantages[t] = delta + gae_decay * next_valid_nonterminal * last_valid_gae

            # Update pointers for the next backward step
            last_valid_gae = advantages[t]
            next_valid_value = self.values[t]
            next_valid_nonterminal = 1.0 - self.dones[t]
            current_gap = 1  # Reset gap since we hit an active step

        returns = advantages + self.values
        return advantages, returns
