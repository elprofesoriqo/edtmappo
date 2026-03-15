import numpy as np


class AsynchronousMPEWrapper:
    """
    A custom wrapper around pettingzoo.mpe.simple_tag_v3 to support
    Epistemic Time-Dilation (ETD) / Dynamic Frame-Skipping.
    Standardizes the API to match LBF/GRF wrappers.
    """

    def __init__(self, env):
        self.env = env
        self.possible_agents = env.possible_agents

        # Internal state for ETD
        self.sleep_timers = {agent: 0 for agent in self.possible_agents}
        self.last_actions = {agent: None for agent in self.possible_agents}

    def reset(self, seed=None, options=None):
        """Reset the environment and the ETD internal state."""
        self.sleep_timers = {agent: 0 for agent in self.possible_agents}
        self.last_actions = {agent: None for agent in self.possible_agents}

        obs_dict, _ = self.env.reset(seed=seed, options=options)

        # Create global state (concatenated observations)
        # Sort keys to ensure deterministic order
        sorted_agents = sorted(self.possible_agents)
        global_state = np.concatenate([obs_dict[a] for a in sorted_agents])

        avail_actions = {a: np.ones(self.env.action_space(a).n) for a in self.possible_agents}

        return obs_dict, global_state, avail_actions

    @property
    def agents(self):
        return self.env.agents

    def step(self, actions_dict, sleep_ticks_dict=None):
        """
        Steps the underlying MPE Parallel Env.
        Returns: (obs, global_state, rewards, dones, avail_actions, active_masks)
        """
        if sleep_ticks_dict is None:
            sleep_ticks_dict = {agent: 0 for agent in self.agents}

        executable_actions = {}
        active_masks = {}

        for agent in self.agents:
            if self.sleep_timers[agent] > 0:
                # Agent is sleeping: repeat last action, decrement timer, mask it out
                executable_actions[agent] = self.last_actions[agent]
                self.sleep_timers[agent] -= 1
                active_masks[agent] = 0.0
            else:
                # Agent is awake: take new action, set sleep timer, keep active
                if agent in actions_dict:
                    executable_actions[agent] = actions_dict[agent]
                    self.last_actions[agent] = actions_dict[agent]

                    # Store sleep duration requested by the neural network
                    self.sleep_timers[agent] = sleep_ticks_dict.get(agent, 0)
                    active_masks[agent] = 1.0
                else:
                    # Fallback (safety for PettingZoo step structure issues)
                    # Use 0 or random valid action
                    executable_actions[agent] = 0
                    active_masks[agent] = 0.0

        # Pass the processed actions to the underlying environment
        obs_dict, rewards_dict, terminations, truncations, infos = self.env.step(executable_actions)

        # Construct global state
        sorted_agents = sorted(self.possible_agents)
        # Handle case where agents might die/disappear from obs
        global_obs_list = []
        for a in sorted_agents:
            if a in obs_dict:
                global_obs_list.append(obs_dict[a])
            else:
                # If agent is dead/gone, append zeros of correct shape
                obs_shape = self.env.observation_space(a).shape
                global_obs_list.append(np.zeros(obs_shape, dtype=np.float32))

        global_state = np.concatenate(global_obs_list)

        # Construct dones dict
        dones_dict = {a: terminations.get(a, False) or truncations.get(a, False) for a in self.possible_agents}

        # Construct available actions
        avail_actions = {a: np.ones(self.env.action_space(a).n) for a in self.possible_agents}

        # Return standardized tuple
        return obs_dict, global_state, rewards_dict, dones_dict, avail_actions, active_masks
