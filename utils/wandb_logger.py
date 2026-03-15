import time

import numpy as np
import torch

import wandb


class AdaptiveEntropyScheduler:
    """
    Adaptive Entropy Threshold Scheduler.

    Linearly anneals the entropy threshold from an initial low value (forcing
    exploration) to a final high value (allowing efficiency) over the course
    of training.
    """

    def __init__(self, initial_threshold=0.5, final_threshold=1.5, total_updates=1000):
        self.initial_threshold = initial_threshold
        self.final_threshold = final_threshold
        self.total_updates = total_updates
        self.current_update = 0

    def step(self):
        """Advance the scheduler by one step."""
        self.current_update += 1

    def get_threshold(self):
        """Get the current entropy threshold."""
        fraction = min(1.0, self.current_update / self.total_updates)
        current = self.initial_threshold + fraction * (self.final_threshold - self.initial_threshold)
        return current


class PerformanceTracker:
    """
    Performance tracker for ETD-MAPPO experiments.

    Tracks win rates, rewards, and temporal specialization metrics.
    """

    def __init__(self, agents, window_size=100):
        self.agents = agents
        self.window_size = window_size

        # Episode history
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []

        # Per-agent history
        self.agent_sleep_rates = {a: [] for a in agents}
        self.agent_entropies = {a: [] for a in agents}
        self.agent_rewards = {a: [] for a in agents}

        # Current episode accumulators
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.current_agent_sleeps = {a: 0 for a in agents}
        self.current_agent_steps = {a: 0 for a in agents}
        self.current_agent_entropies = {a: [] for a in agents}
        self.current_agent_rewards = {a: 0.0 for a in agents}

    def step(self, rewards_dict, active_masks, entropies_dict=None):
        """Record a single step."""
        self.current_episode_length += 1

        for a in self.agents:
            reward = rewards_dict.get(a, 0.0)
            mask = active_masks.get(a, 1.0)

            self.current_agent_rewards[a] += reward
            self.current_episode_reward += reward
            self.current_agent_steps[a] += 1

            if mask == 0.0:
                self.current_agent_sleeps[a] += 1

            if entropies_dict and a in entropies_dict:
                self.current_agent_entropies[a].append(entropies_dict[a])

    def end_episode(self, success=None):
        """Finalize the current episode."""
        self.episode_rewards.append(self.current_episode_reward)
        self.episode_lengths.append(self.current_episode_length)

        if success is not None:
            self.episode_successes.append(1.0 if success else 0.0)

        for a in self.agents:
            if self.current_agent_steps[a] > 0:
                sleep_rate = self.current_agent_sleeps[a] / self.current_agent_steps[a]
                self.agent_sleep_rates[a].append(sleep_rate)

            if self.current_agent_entropies[a]:
                avg_entropy = np.mean(self.current_agent_entropies[a])
                self.agent_entropies[a].append(avg_entropy)

            self.agent_rewards[a].append(self.current_agent_rewards[a])

        # Reset accumulators
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.current_agent_sleeps = {a: 0 for a in self.agents}
        self.current_agent_steps = {a: 0 for a in self.agents}
        self.current_agent_entropies = {a: [] for a in self.agents}
        self.current_agent_rewards = {a: 0.0 for a in self.agents}

    def get_metrics(self):
        """Get aggregated metrics for logging."""
        metrics = {}

        if self.episode_rewards:
            recent_rewards = self.episode_rewards[-self.window_size :]
            metrics['performance/episode_reward_mean'] = np.mean(recent_rewards)
            metrics['performance/episode_reward_std'] = np.std(recent_rewards)

        if self.episode_lengths:
            recent_lengths = self.episode_lengths[-self.window_size :]
            metrics['performance/episode_length_mean'] = np.mean(recent_lengths)

        if self.episode_successes:
            recent_successes = self.episode_successes[-self.window_size :]
            metrics['performance/win_rate'] = np.mean(recent_successes) * 100

        for a in self.agents:
            if self.agent_sleep_rates[a]:
                recent_sleep = self.agent_sleep_rates[a][-self.window_size :]
                metrics[f'temporal/{a}_sleep_rate'] = np.mean(recent_sleep) * 100

            if self.agent_entropies[a]:
                recent_ent = self.agent_entropies[a][-self.window_size :]
                metrics[f'temporal/{a}_avg_entropy'] = np.mean(recent_ent)

            if self.agent_rewards[a]:
                recent_rew = self.agent_rewards[a][-self.window_size :]
                metrics[f'performance/{a}_reward'] = np.mean(recent_rew)

        return metrics

    def get_temporal_specialization_summary(self):
        """Get per-agent temporal specialization stats."""
        summary = {}

        for a in self.agents:
            if self.agent_sleep_rates[a]:
                summary[a] = {
                    'avg_sleep_rate': np.mean(self.agent_sleep_rates[a][-self.window_size :]) * 100,
                    'sleep_rate_std': np.std(self.agent_sleep_rates[a][-self.window_size :]) * 100,
                    'avg_entropy': np.mean(self.agent_entropies[a][-self.window_size :])
                    if self.agent_entropies[a]
                    else 0.0,
                }

        return summary


def log_episode_timeline(env, actors, entropy_threshold, max_sleep_ticks, device, hidden_size=64, episode='Eval'):
    """
    Runs a deterministic episode and records tick rates and entropy for
    every frame to visualize the dynamic frame-skipping behavior.
    """
    obs_dict, global_state, _ = env.reset()
    agents = env.possible_agents

    timeline_data = {agent: {'tick_rate': [], 'entropy': [], 'action': [], 'reward': []} for agent in agents}

    max_frames = 150

    hidden_states = {a: torch.zeros((1, hidden_size)).to(device) for a in agents}

    next_obs = {a: torch.tensor(obs_dict[a], dtype=torch.float32).to(device) for a in agents}

    total_reward = 0.0

    print(f"[*] Extracting 'Money Shot' Timeline tracking for Episode ({episode})...")

    with torch.no_grad():
        for _frame in range(max_frames):
            actions_dict = {}
            sleep_ticks_dict = {}

            for a in agents:
                active_mask = torch.ones((1, 1)).to(device)

                if env.sleep_timers[a] > 0:
                    # Agent is asleep
                    timeline_data[a]['tick_rate'].append(0.0)
                    last_ent = timeline_data[a]['entropy'][-1] if len(timeline_data[a]['entropy']) > 0 else 0.0
                    timeline_data[a]['entropy'].append(last_ent)
                    timeline_data[a]['action'].append(-1)

                    actions_dict[a] = env.last_actions[a]
                    sleep_ticks_dict[a] = 0
                    continue

                # Agent is awake
                action, _, entropy, sleep_ticks, new_hidden = actors[a].get_action_and_sleep_ticks(
                    next_obs[a].unsqueeze(0),
                    hidden_states[a],
                    active_mask,
                    entropy_threshold=entropy_threshold,
                    max_sleep_ticks=max_sleep_ticks,
                )

                hidden_states[a] = new_hidden

                timeline_data[a]['tick_rate'].append(1.0)
                timeline_data[a]['entropy'].append(entropy.item())
                timeline_data[a]['action'].append(action.item())

                actions_dict[a] = action.item()
                sleep_ticks_dict[a] = sleep_ticks.item() if isinstance(sleep_ticks, torch.Tensor) else sleep_ticks

            next_obs_dict, next_global_state, rewards_dict, dones_dict, _, active_masks = env.step(
                actions_dict, sleep_ticks_dict
            )

            for a in agents:
                reward = rewards_dict.get(a, 0.0)
                timeline_data[a]['reward'].append(reward)
                total_reward += reward

                if a in next_obs_dict:
                    next_obs[a] = torch.tensor(next_obs_dict[a], dtype=torch.float32).to(device)

            if all(dones_dict.values()):
                break

    # Format for WandB
    data_rows = []
    frames = max(len(timeline_data[a]['tick_rate']) for a in agents)

    for frame_idx in range(frames):
        row = [frame_idx]
        for a in agents:
            if frame_idx < len(timeline_data[a]['tick_rate']):
                row.append(timeline_data[a]['tick_rate'][frame_idx])
                row.append(timeline_data[a]['entropy'][frame_idx])
                row.append(timeline_data[a]['reward'][frame_idx])
            else:
                row.append(0.0)
                row.append(0.0)
                row.append(0.0)
        data_rows.append(row)

    columns = ['Frame']
    for a in agents:
        columns.extend([f'{a}_TickRate', f'{a}_Entropy', f'{a}_Reward'])

    table = wandb.Table(data=data_rows, columns=columns)
    wandb.log({f'MoneyShot_Timeline_{episode}': table})

    for a in agents:
        if timeline_data[a]['tick_rate']:
            awake_rate = np.mean(timeline_data[a]['tick_rate']) * 100
            wandb.log({f'timeline/{a}_awake_rate_{episode}': awake_rate})

    print(f'[*] Timeline logged successfully to WandB! Total reward: {total_reward:.2f}')

    return timeline_data, total_reward


def generate_money_shot_figure(
    timeline_data, agents, save_path=None, title='ETD-MAPPO Money Shot: Entropy vs Tick Rate'
):
    """
    Generate a publication-ready Money Shot figure.
    """
    try:
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not available for figure generation')
        return None

    n_agents = len(agents)
    fig, axes = plt.subplots(n_agents, 1, figsize=(14, 3 * n_agents), sharex=True)

    if n_agents == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, n_agents))

    for idx, (a, ax) in enumerate(zip(agents, axes)):
        frames = np.arange(len(timeline_data[a]['tick_rate']))
        tick_rates = np.array(timeline_data[a]['tick_rate'])
        entropies = np.array(timeline_data[a]['entropy'])

        ax2 = ax.twinx()
        ax2.plot(frames, entropies, color=colors[idx], linewidth=2, label=f'{a} Entropy')
        ax2.set_ylabel('Policy Entropy $\\mathcal{H}$', color=colors[idx])
        ax2.tick_params(axis='y', labelcolor=colors[idx])
        ax2.set_ylim(0, max(entropies) * 1.2 if len(entropies) > 0 and max(entropies) > 0 else 2.0)

        for i, tr in enumerate(tick_rates):
            color = '#90EE90' if tr == 1.0 else '#FFB6C1'
            ax.axvspan(i, i + 1, alpha=0.3, color=color)

        ax.set_ylabel('Tick Rate')
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Asleep', 'Awake'])
        ax.set_title(f'{a}: Temporal Dynamics', fontsize=12, fontweight='bold')

        awake_patch = mpatches.Patch(color='#90EE90', alpha=0.3, label='Awake (Inference)')
        asleep_patch = mpatches.Patch(color='#FFB6C1', alpha=0.3, label='Asleep (Skip)')
        ax.legend(handles=[awake_patch, asleep_patch], loc='upper left')

    axes[-1].set_xlabel('Environment Frame', fontsize=12)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)  # Release file handle
            time.sleep(0.5)  # Wait for file system flush
            print(f'[*] Money Shot figure saved to {save_path}')
            wandb.log({'MoneyShot_Figure': wandb.Image(save_path)})
        except Exception as e:
            print(f'[!] Error saving/logging figure: {e}')
    else:
        plt.close(fig)

    return None


def log_temporal_specialization(tracker, update_num):
    """
    Log temporal specialization charts to WandB.
    """
    summary = tracker.get_temporal_specialization_summary()

    if not summary:
        return

    agents = list(summary.keys())
    sleep_rates = [summary[a]['avg_sleep_rate'] for a in agents]
    entropies = [summary[a]['avg_entropy'] for a in agents]

    data = [[a, sr, ent] for a, sr, ent in zip(agents, sleep_rates, entropies)]
    table = wandb.Table(data=data, columns=['Agent', 'Sleep Rate (%)', 'Avg Entropy'])

    wandb.log(
        {
            f'temporal_specialization/sleep_rates_update_{update_num}': wandb.plot.bar(
                table, 'Agent', 'Sleep Rate (%)', title=f'Per-Agent Sleep Rates (Update {update_num})'
            ),
            f'temporal_specialization/entropies_update_{update_num}': wandb.plot.bar(
                table, 'Agent', 'Avg Entropy', title=f'Per-Agent Average Entropy (Update {update_num})'
            ),
        }
    )

    for a in agents:
        wandb.log(
            {
                f'specialization/{a}_sleep_rate': summary[a]['avg_sleep_rate'],
                f'specialization/{a}_entropy': summary[a]['avg_entropy'],
            }
        )

    sleep_rate_variance = np.var(sleep_rates)
    wandb.log({'specialization/cross_agent_variance': sleep_rate_variance})

    print(f'[*] Temporal Specialization Analysis (Update {update_num}):')
    for a in agents:
        print(
            f'    {a}: Sleep Rate = {summary[a]["avg_sleep_rate"]:.1f}%, Avg Entropy = {summary[a]["avg_entropy"]:.3f}'
        )


def run_evaluation_episode(env, actors, entropy_threshold, max_sleep_ticks, device, hidden_size=64, deterministic=True):
    """
    Run a single evaluation episode.
    """
    obs_dict, global_state, _ = env.reset()
    agents = env.possible_agents

    hidden_states = {a: torch.zeros((1, hidden_size)).to(device) for a in agents}
    next_obs = {a: torch.tensor(obs_dict[a], dtype=torch.float32).to(device) for a in agents}

    total_reward = 0.0
    agent_rewards = {a: 0.0 for a in agents}
    agent_sleeps = {a: 0 for a in agents}
    agent_steps = {a: 0 for a in agents}
    episode_length = 0
    max_frames = 200

    with torch.no_grad():
        for _frame in range(max_frames):
            actions_dict = {}
            sleep_ticks_dict = {}

            for a in agents:
                agent_steps[a] += 1
                active_mask = torch.ones((1, 1)).to(device)

                if env.sleep_timers[a] > 0:
                    actions_dict[a] = env.last_actions[a]
                    sleep_ticks_dict[a] = 0
                    agent_sleeps[a] += 1
                    continue

                action, _, entropy, sleep_ticks, new_hidden = actors[a].get_action_and_sleep_ticks(
                    next_obs[a].unsqueeze(0),
                    hidden_states[a],
                    active_mask,
                    entropy_threshold=entropy_threshold,
                    max_sleep_ticks=max_sleep_ticks,
                )

                hidden_states[a] = new_hidden

                if deterministic:
                    logits, _ = actors[a].forward(next_obs[a].unsqueeze(0), hidden_states[a], active_mask)
                    action = logits.argmax(dim=-1)

                actions_dict[a] = action.item() if isinstance(action, torch.Tensor) else action
                sleep_ticks_dict[a] = sleep_ticks.item() if isinstance(sleep_ticks, torch.Tensor) else sleep_ticks

            next_obs_dict, _, rewards_dict, dones_dict, _, _ = env.step(actions_dict, sleep_ticks_dict)

            episode_length += 1

            for a in agents:
                reward = rewards_dict.get(a, 0.0)
                agent_rewards[a] += reward
                total_reward += reward

                if a in next_obs_dict:
                    next_obs[a] = torch.tensor(next_obs_dict[a], dtype=torch.float32).to(device)

            if all(dones_dict.values()):
                break

    agent_sleep_rates = {a: agent_sleeps[a] / agent_steps[a] if agent_steps[a] > 0 else 0.0 for a in agents}

    return {
        'total_reward': total_reward,
        'episode_length': episode_length,
        'agent_rewards': agent_rewards,
        'agent_sleep_rates': agent_sleep_rates,
        'success': total_reward > 0,
    }


def run_baseline_comparison(env, actors, device, hidden_size=64, num_eval_episodes=10):
    """
    Compare ETD, Vanilla, and Fixed-Skip modes.
    """
    results = {
        'etd': {'rewards': [], 'sleep_rates': [], 'successes': []},
        'vanilla': {'rewards': [], 'sleep_rates': [], 'successes': []},
        'fixed_skip': {'rewards': [], 'sleep_rates': [], 'successes': []},
    }

    agents = env.possible_agents
    max_sleep = 3

    for _ep in range(num_eval_episodes):
        # ETD Mode
        etd_result = run_evaluation_episode(
            env, actors, entropy_threshold=1.5, max_sleep_ticks=max_sleep, device=device, hidden_size=hidden_size
        )
        results['etd']['rewards'].append(etd_result['total_reward'])
        results['etd']['sleep_rates'].append(np.mean(list(etd_result['agent_sleep_rates'].values())))
        results['etd']['successes'].append(etd_result['success'])

        # Vanilla Mode
        vanilla_result = run_evaluation_episode(
            env, actors, entropy_threshold=-1.0, max_sleep_ticks=0, device=device, hidden_size=hidden_size
        )
        results['vanilla']['rewards'].append(vanilla_result['total_reward'])
        results['vanilla']['sleep_rates'].append(0.0)
        results['vanilla']['successes'].append(vanilla_result['success'])

        # Fixed-Skip Mode
        fixed_result = run_evaluation_episode(
            env, actors, entropy_threshold=999.0, max_sleep_ticks=max_sleep, device=device, hidden_size=hidden_size
        )
        results['fixed_skip']['rewards'].append(fixed_result['total_reward'])
        results['fixed_skip']['sleep_rates'].append(np.mean(list(fixed_result['agent_sleep_rates'].values())))
        results['fixed_skip']['successes'].append(fixed_result['success'])

    summary = {}
    for mode in ['etd', 'vanilla', 'fixed_skip']:
        summary[mode] = {
            'mean_reward': np.mean(results[mode]['rewards']),
            'std_reward': np.std(results[mode]['rewards']),
            'win_rate': np.mean(results[mode]['successes']) * 100,
            'mean_sleep_rate': np.mean(results[mode]['sleep_rates']) * 100,
            'flop_reduction': np.mean(results[mode]['sleep_rates']) * 100,
        }

    return summary, results


def log_comparison_table(summary, update_num):
    """Log comparison table to WandB."""
    data = []
    for mode in ['vanilla', 'fixed_skip', 'etd']:
        data.append(
            [
                mode.upper(),
                f'{summary[mode]["mean_reward"]:.2f} ± {summary[mode]["std_reward"]:.2f}',
                f'{summary[mode]["win_rate"]:.1f}%',
                f'{summary[mode]["flop_reduction"]:.1f}%',
            ]
        )

    table = wandb.Table(data=data, columns=['Method', 'Reward (mean ± std)', 'Win Rate', 'FLOP Reduction'])

    wandb.log({f'comparison/baseline_table_update_{update_num}': table})

    for mode in ['vanilla', 'fixed_skip', 'etd']:
        wandb.log(
            {
                f'comparison/{mode}_reward': summary[mode]['mean_reward'],
                f'comparison/{mode}_win_rate': summary[mode]['win_rate'],
                f'comparison/{mode}_flop_reduction': summary[mode]['flop_reduction'],
            }
        )

    print(f'\n[*] Baseline Comparison (Update {update_num}):')
    print(f'    {"Method":<12} {"Reward":<20} {"Win Rate":<12} {"FLOP Reduction":<15}')
    print(f'    {"-" * 60}')
    for mode in ['vanilla', 'fixed_skip', 'etd']:
        print(
            f'    {mode.upper():<12} {summary[mode]["mean_reward"]:>6.2f} ± {summary[mode]["std_reward"]:<6.2f}   '
            f'{summary[mode]["win_rate"]:>6.1f}%      {summary[mode]["flop_reduction"]:>6.1f}%'
        )
