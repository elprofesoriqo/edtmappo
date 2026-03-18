try:
    import gfootball.env as football_env
    from gfootball.env import football_env as football_env_core

    if not hasattr(football_env_core.FootballEnv, '_is_patched'):
        _original_reset = football_env_core.FootballEnv.reset
        def _patched_reset(self, **kwargs):
            obs = _original_reset(self)
            return obs, {}
        football_env_core.FootballEnv.reset = _patched_reset

        _original_step = football_env_core.FootballEnv.step
        def _patched_step(self, action):
            obs, reward, done, info = _original_step(self, action)
            return obs, reward, done, False, info
        football_env_core.FootballEnv.step = _patched_step

        football_env_core.FootballEnv._is_patched = True

    GFOOTBALL_AVAILABLE = True
except ImportError:
    GFOOTBALL_AVAILABLE = False
    print('Warning: gfootball not installed. Using Mock GRF Environment for verification.')

import numpy as np


class MockGRFEnv:
    """
    Mock environment for Google Research Football to allow code verification
    without heavy binary dependencies on Windows.
    """

    def __init__(self, scenario_name, number_of_left_players_agent_controls=3):
        self.scenario_name = scenario_name
        self.num_agents = number_of_left_players_agent_controls

    def reset(self):
        # Return obs only (old gym API style)
        return np.zeros((self.num_agents, 115))

    def step(self, actions):
        # Return (obs, reward, done, info) - old gym API
        obs = np.zeros((self.num_agents, 115))
        rewards = np.zeros(self.num_agents)
        done = False
        info = {}
        return obs, rewards, done, info


class AsynchronousGRFWrapper:
    """
    Wrapper for Google Research Football (GRF).
    Handles both old gym API (reset returns obs) and new gym API (reset returns obs, info).
    """

    def __init__(self, env_name='academy_3_vs_1_with_keeper', number_of_left_players_agent_controls=3):
        self.n_agents = number_of_left_players_agent_controls

        if GFOOTBALL_AVAILABLE:
            self.env = football_env.create_environment(
                env_name=env_name,
                stacked=False,
                representation='simple115v2',
                rewards='scoring,checkpoints',
                write_goal_dumps=False,
                write_full_episode_dumps=False,
                render=False,
                number_of_left_players_agent_controls=self.n_agents,
            )
            self.using_real_grf = True
        else:
            self.env = MockGRFEnv(env_name, number_of_left_players_agent_controls=self.n_agents)
            self.using_real_grf = False

        self.possible_agents = [f'agent_{i}' for i in range(self.n_agents)]

        # Dimensions
        self.obs_dim = 115
        self.n_actions = 19

        # Asynchronous state
        self.sleep_timers = {a: 0 for a in self.possible_agents}
        self.last_actions = {a: 0 for a in self.possible_agents}

    def _to_dict(self, list_data):
        """Converts GRF's list outputs into PettingZoo dictionary formats."""
        return {self.possible_agents[i]: list_data[i] for i in range(self.n_agents)}

    def _extract_obs(self, reset_result):
        """
        Handle different gym API versions.
        Old API: reset() returns obs
        New API: reset() returns (obs, info)
        """
        if isinstance(reset_result, tuple):
            return reset_result[0]
        return reset_result

    def reset(self):
        reset_result = self.env.reset()
        obs = self._extract_obs(reset_result)

        # Ensure obs is the right shape
        if obs.ndim == 1:
            # Single agent case or flattened - reshape
            obs = obs.reshape(self.n_agents, -1)

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
                action = actions_dict.get(a, 0)
                sleep_ticks = sleep_ticks_dict.get(a, 0)

                self.last_actions[a] = action
                self.sleep_timers[a] = sleep_ticks
                active_masks[a] = 1.0
                env_actions.append(action)

        # Step the underlying synchronous GRF environment
        step_result = self.env.step(env_actions)

        # Handle different gym API versions
        # API: (obs, reward, terminated, truncated, info)
        if len(step_result) == 4:
            next_obs, rewards, done, infos = step_result
        elif len(step_result) == 5:
            next_obs, rewards, terminated, truncated, infos = step_result
            done = terminated or truncated
        else:
            raise ValueError(f'Unexpected step result length: {len(step_result)}')

        # Ensure obs is the right shape
        if next_obs.ndim == 1:
            next_obs = next_obs.reshape(self.n_agents, -1)

        obs_dict = self._to_dict(next_obs)
        global_state = np.concatenate(next_obs)

        # Handle rewards - could be scalar or array
        if np.isscalar(rewards):
            rewards = np.array([rewards] * self.n_agents)
        elif isinstance(rewards, (list, np.ndarray)) and len(rewards) != self.n_agents:
            # If rewards shape doesn't match, broadcast
            rewards = np.full(self.n_agents, np.mean(rewards))

        rewards_dict = self._to_dict(rewards)

        # Broadcast done
        dones_dict = {a: done for a in self.possible_agents}
        avail_actions = {a: np.ones(self.n_actions) for a in self.possible_agents}

        return obs_dict, global_state, rewards_dict, dones_dict, avail_actions, active_masks
