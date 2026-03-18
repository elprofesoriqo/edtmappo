import gymnasium as gym
import numpy as np


class AsynchronousLBFWrapper:
    """
    A wrapper for the Level-Based Foraging (LBF) environment to support ETD-MAPPO asynchronous execution.
    Converts the standard OpenAI Gym tuple interface into a dictionary-based PettingZoo style interface.
    Handles 'sleep_timers' internally, repeating last known actions for dormant agents.

    LBF natively supports partial observability (grid vision) and heterogeneous cooperation.
    """

    def __init__(self, env_id='Foraging-8x8-2p-2f-v3'):
        # LBF registers itself into standard gym
        self.env = gym.make(env_id, disable_env_checker=True)

        self.n_agents = len(self.env.observation_space)
        self.possible_agents = [f'agent_{i}' for i in range(self.n_agents)]

        # Dimensions
        # LBF observations are flat vectors representing the grid and player stats
        self.obs_dim = self.env.observation_space[0].shape[0]
        self.n_actions = self.env.action_space[0].n

        # Asynchronous state
        self.sleep_timers = {a: 0 for a in self.possible_agents}
        self.last_actions = {a: 0 for a in self.possible_agents}  # 0 is typically NO-OP in LBF

    def _to_dict(self, list_data):
        """Converts LBF's native list outputs into PettingZoo dictionary formats."""
        return {self.possible_agents[i]: list_data[i] for i in range(self.n_agents)}

    def reset(self):
        obs, _ = self.env.reset()

        self.sleep_timers = {a: 0 for a in self.possible_agents}
        self.last_actions = {a: 0 for a in self.possible_agents}

        obs_dict = self._to_dict(obs)
        global_state = np.concatenate(obs)
        avail_actions = {a: np.ones(self.n_actions) for a in self.possible_agents}

        return obs_dict, global_state, avail_actions

    def step(self, actions_dict, sleep_ticks_dict):
        """
        Executes an asynchronous step.
        If an agent is asleep, its provided action is ignored and it repeats its last action.
        """
        active_masks = {}
        env_actions = []

        for _i, a in enumerate(self.possible_agents):
            if self.sleep_timers[a] > 0:
                # Agent is Asleep (Time Dilation)
                self.sleep_timers[a] -= 1
                active_masks[a] = 0.0
                env_actions.append(self.last_actions[a])
            else:
                # Agent is Awake
                action = actions_dict[a]
                sleep_ticks = sleep_ticks_dict[a]

                self.last_actions[a] = action
                self.sleep_timers[a] = sleep_ticks
                active_masks[a] = 1.0
                env_actions.append(action)

        # Step the underlying synchronous LBF environment
        # Gymnasium returns 5 values: obs, reward, terminated, truncated, info
        next_obs, rewards, terminated, truncated, infos = self.env.step(env_actions)

        obs_dict = self._to_dict(next_obs)
        global_state = np.concatenate(next_obs)
        rewards_dict = self._to_dict(rewards)

        # LBF returns a single global boolean for termination/truncation
        # We broadcast this to all agents
        done = terminated or truncated
        dones_dict = {a: done for a in self.possible_agents}

        avail_actions = {a: np.ones(self.n_actions) for a in self.possible_agents}

        return obs_dict, global_state, rewards_dict, dones_dict, avail_actions, active_masks

    def get_avail_actions(self):
        return [np.ones(self.n_actions) for _ in range(self.n_agents)]

    def get_obs(self):
        # We don't strictly need this due to the comprehensive reset/step returns,
        # but included for SMAC API compatibility.
        pass
