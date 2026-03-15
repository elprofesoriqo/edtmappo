"""
ETD-MAPPO Training Script for Level-Based Foraging (LBF)

This script implements comprehensive tracking for:
- Win Rate / Task Performance
- Money Shot Timeline Visualization
- Per-Agent Temporal Specialization
- Baseline Comparison (Vanilla vs Fixed-Skip vs ETD)
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.optim as optim

import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from algorithms.ppo.buffer import AsynchronousRolloutBuffer
from algorithms.ppo.loss import compute_masked_ppo_loss
from envs.lbf_wrapper import AsynchronousLBFWrapper
from models.rnn_networks import AsynchronousGRUActor, AsynchronousGRUCritic
from utils.wandb_logger import (
    AdaptiveEntropyScheduler,
    PerformanceTracker,
    generate_money_shot_figure,
    log_comparison_table,
    log_episode_timeline,
    log_temporal_specialization,
    run_baseline_comparison,
)


def parse_args():
    parser = argparse.ArgumentParser(description='ETD-MAPPO LBF Comprehensive Evaluator')
    parser.add_argument('--mode', type=str, default='etd', choices=['etd', 'vanilla', 'fixed_skip'])
    parser.add_argument('--fixed_skip_amount', type=int, default=3)
    parser.add_argument('--map_name', type=str, default='Foraging-8x8-3p-3f-v3')
    parser.add_argument('--num_envs', type=int, default=1)
    parser.add_argument('--num_updates', type=int, default=1000)
    parser.add_argument('--eval_interval', type=int, default=100, help='Run evaluation every N updates')
    parser.add_argument('--save_figures', action='store_true', help='Save Money Shot figures locally')
    return parser.parse_args()


def determine_success(rewards_dict, episode_length):
    """
    Determine if an LBF episode was successful.
    Success = collected at least one food item (positive total reward).
    """
    total_reward = sum(rewards_dict.values()) if isinstance(rewards_dict, dict) else rewards_dict
    return total_reward > 0


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Initializing LBF '{args.map_name}' on device: {device} | Mode: {args.mode.upper()}")

    # Create output directory for figures
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    wandb.init(project='ETD-MAPPO-Research', name=f'LBF-{args.map_name}-{args.mode}-comprehensive', config=vars(args))

    env = AsynchronousLBFWrapper(env_id=args.map_name)

    # Hyperparameters
    num_steps = 100
    num_updates = args.num_updates
    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.2
    ent_coef = 0.01
    vf_coef = 0.5
    hidden_size = 64

    max_sleep_ticks = args.fixed_skip_amount if args.mode == 'fixed_skip' else 3

    # Mode-specific entropy scheduler
    if args.mode == 'vanilla':
        entropy_scheduler = AdaptiveEntropyScheduler(
            initial_threshold=-1.0, final_threshold=-1.0, total_updates=num_updates
        )
    elif args.mode == 'fixed_skip':
        entropy_scheduler = AdaptiveEntropyScheduler(
            initial_threshold=999.0, final_threshold=999.0, total_updates=num_updates
        )
    else:
        # ETD: Start conservative, gradually allow more sleep
        entropy_scheduler = AdaptiveEntropyScheduler(
            initial_threshold=0.5, final_threshold=1.75, total_updates=num_updates
        )

    agents = env.possible_agents
    obs_dim = env.obs_dim
    global_obs_dim = obs_dim * len(agents)
    act_dim = env.n_actions

    # Initialize networks
    actors = {}
    critics = {}
    optimizers = {}
    buffers = {}

    for a in agents:
        actors[a] = AsynchronousGRUActor(obs_dim, act_dim, hidden_size).to(device)
        critics[a] = AsynchronousGRUCritic(global_obs_dim, hidden_size).to(device)

        params = list(actors[a].parameters()) + list(critics[a].parameters())
        optimizers[a] = optim.Adam(params, lr=5e-4, eps=1e-5)

        buffers[a] = AsynchronousRolloutBuffer(num_steps, obs_dim, (), device)

    print(f'[*] Networks constructed for {len(agents)} LBF agents.')

    # Initialize performance tracker
    tracker = PerformanceTracker(agents)

    batch_global_obs = torch.zeros((num_steps, global_obs_dim)).to(device)

    # Training loop
    for update in range(1, num_updates + 1):
        current_threshold = entropy_scheduler.get_threshold()
        entropy_scheduler.step()

        # Reset buffers
        for a in agents:
            buffers[a].reset()

        obs_dicts, global_states, avail_dicts = env.reset()
        next_obs = {a: torch.tensor(obs_dicts[a], dtype=torch.float32).to(device) for a in agents}

        actor_hidden = {a: torch.zeros((1, hidden_size)).to(device) for a in agents}
        critic_hidden = {a: torch.zeros((1, hidden_size)).to(device) for a in agents}

        # Tracking variables
        saved_inferences = 0
        total_evaluations = 0
        episode_rewards_sum = {a: 0.0 for a in agents}
        episodes_completed = 0

        # Rollout phase
        for step in range(num_steps):
            total_evaluations += len(agents)

            global_obs = torch.tensor(global_states, dtype=torch.float32).to(device)
            batch_global_obs[step] = global_obs

            actions_dict = {}
            sleep_ticks_dict = {}
            step_values = {}
            step_logprobs = {}
            step_entropies = {}

            with torch.no_grad():
                for a in agents:
                    active_mask_tensor = torch.ones((1, 1)).to(device)

                    action, log_prob, entropy, sleep_ticks, new_actor_h = actors[a].get_action_and_sleep_ticks(
                        next_obs[a].unsqueeze(0),
                        actor_hidden[a],
                        active_mask_tensor,
                        entropy_threshold=current_threshold,
                        max_sleep_ticks=max_sleep_ticks,
                    )

                    value, new_critic_h = critics[a](global_obs.unsqueeze(0), critic_hidden[a], active_mask_tensor)

                    actor_hidden[a] = new_actor_h
                    critic_hidden[a] = new_critic_h

                    step_values[a] = value.squeeze()
                    step_logprobs[a] = log_prob.squeeze()
                    step_entropies[a] = entropy.item()

                    # Mode-specific sleep logic
                    if args.mode == 'vanilla':
                        final_sleep = 0
                    elif args.mode == 'fixed_skip':
                        final_sleep = max_sleep_ticks
                    else:
                        final_sleep = sleep_ticks.item() if isinstance(sleep_ticks, torch.Tensor) else sleep_ticks

                    actions_dict[a] = action.item()
                    sleep_ticks_dict[a] = final_sleep

            # Step environment
            next_obs_dict, next_global_state, rewards_dict, dones_dict, next_avail_dicts, active_masks = env.step(
                actions_dict, sleep_ticks_dict
            )

            # Track performance
            tracker.step(rewards_dict, active_masks, step_entropies)

            for a in agents:
                mask_val = active_masks.get(a, 0.0)
                reward = rewards_dict.get(a, 0.0)
                done = dones_dict.get(a, False)

                episode_rewards_sum[a] += reward

                if mask_val == 0.0:
                    saved_inferences += 1

                buffers[a].insert_step(
                    obs=next_obs[a],
                    action=torch.tensor(actions_dict[a]).to(device),
                    logprob=step_logprobs[a],
                    value=step_values[a],
                    reward=reward,
                    done=torch.tensor(done, dtype=torch.float32).to(device),
                    active_mask=mask_val,
                )

                next_obs[a] = torch.tensor(next_obs_dict[a], dtype=torch.float32).to(device)

            global_states = next_global_state

            # Episode termination
            if all(dones_dict.values()):
                success = determine_success(episode_rewards_sum, step + 1)
                tracker.end_episode(success=success)
                episodes_completed += 1

                # Reset for new episode
                episode_rewards_sum = {a: 0.0 for a in agents}
                obs_dicts, global_states, next_avail_dicts = env.reset()
                next_obs = {a: torch.tensor(obs_dicts[a], dtype=torch.float32).to(device) for a in agents}
                actor_hidden = {a: torch.zeros((1, hidden_size)).to(device) for a in agents}
                critic_hidden = {a: torch.zeros((1, hidden_size)).to(device) for a in agents}

        # PPO Update phase
        with torch.no_grad():
            global_obs = torch.tensor(global_states, dtype=torch.float32).to(device)
            for a in agents:
                dummy_mask = torch.ones((1, 1)).to(device)
                next_value, _ = critics[a](global_obs.unsqueeze(0), critic_hidden[a], dummy_mask)
                next_value = next_value.squeeze()

                advantages, returns = buffers[a].compute_gae(next_value, buffers[a].dones[-1], gamma, gae_lambda)

                b_obs = buffers[a].obs
                b_logprobs = buffers[a].logprobs
                b_actions = buffers[a].actions
                b_masks = buffers[a].active_masks

                with torch.enable_grad():
                    b_hidden_a = torch.zeros((num_steps, hidden_size)).to(device)
                    b_active_mask_batch = b_masks.unsqueeze(-1)

                    _, newlogprob, entropy, _, _ = actors[a].get_action_and_sleep_ticks(
                        b_obs, b_hidden_a, b_active_mask_batch, current_threshold, max_sleep_ticks, action=b_actions
                    )

                    b_hidden_c = torch.zeros((num_steps, hidden_size)).to(device)
                    newvalue, _ = critics[a](batch_global_obs, b_hidden_c, b_active_mask_batch)
                    newvalue = newvalue.squeeze()

                    loss, pg_loss, v_loss, ent_mean = compute_masked_ppo_loss(
                        newlogprob,
                        b_logprobs,
                        newvalue,
                        returns,
                        advantages,
                        b_masks,
                        clip_coef,
                        vf_coef,
                        ent_coef,
                        entropy,
                    )

                    optimizers[a].zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(actors[a].parameters()) + list(critics[a].parameters()), 1.0)
                    optimizers[a].step()

        # Compute metrics
        flop_reduction_pct = (saved_inferences / total_evaluations) * 100
        perf_metrics = tracker.get_metrics()

        # Calculate average entropy for this update
        avg_entropy = np.mean([step_entropies[a] for a in agents]) if step_entropies else 0.0

        # Calculate total episode reward for this update
        episode_reward_sum = sum(episode_rewards_sum.values())

        # Log to WandB - standardized metrics across all environments
        log_dict = {
            'compute/FLOP_reduction_pct': flop_reduction_pct,
            'compute/saved_inferences': saved_inferences,
            'dynamics/entropy_threshold': current_threshold,
            'dynamics/avg_policy_entropy': avg_entropy,
            'training/episodes_completed': episodes_completed,
            'training/episode_reward_this_update': episode_reward_sum,
            **perf_metrics,
        }

        # Add per-agent entropy
        for a in agents:
            if a in step_entropies:
                log_dict[f'entropy/{a}'] = step_entropies[a]

        wandb.log(log_dict, step=update)

        if update % args.eval_interval == 0:
            print(f'\n{"=" * 70}')
            print(f'Update {update}/{num_updates} | Mode: {args.mode.upper()}')
            print(f'{"=" * 70}')
            print(f'  FLOP Reduction: {flop_reduction_pct:.1f}%')
            print(f'  Entropy Threshold: {current_threshold:.3f}')
            print(f'  Avg Policy Entropy: {avg_entropy:.3f}')

            if 'performance/win_rate' in perf_metrics:
                print(f'  Win Rate: {perf_metrics["performance/win_rate"]:.1f}%')
            if 'performance/episode_reward_mean' in perf_metrics:
                print(f'  Mean Episode Reward: {perf_metrics["performance/episode_reward_mean"]:.2f}')

            log_temporal_specialization(tracker, update)

            print('\n[*] Generating Money Shot Timeline...')
            timeline_data, total_reward = log_episode_timeline(
                env,
                actors,
                current_threshold,
                max_sleep_ticks,
                device,
                hidden_size=hidden_size,
                episode=f'Update{update}',
            )

            if args.save_figures:
                fig_path = os.path.join(output_dir, f'money_shot_update_{update}.png')
                generate_money_shot_figure(
                    timeline_data,
                    agents,
                    save_path=fig_path,
                    title=f'ETD-MAPPO Money Shot (Update {update}, {args.mode.upper()})',
                )

            # Run baseline comparison every eval_interval * 2
            if update % (args.eval_interval * 2) == 0 and args.mode == 'etd':
                print('\n[*] Running Baseline Comparison...')
                comparison_summary, _ = run_baseline_comparison(
                    env, actors, device, hidden_size=hidden_size, num_eval_episodes=10
                )
                log_comparison_table(comparison_summary, update)

        if update % 50 == 0:
            print(
                f'--- LBF Update {update}/{num_updates} | FLOP: {flop_reduction_pct:.1f}% | Entropy: {avg_entropy:.3f} | Episodes: {episodes_completed} | Reward: {episode_reward_sum:.3f} ---'
            )

    print(f'\n{"=" * 70}')
    print('FINAL EVALUATION')
    print(f'{"=" * 70}')

    timeline_data, total_reward = log_episode_timeline(
        env, actors, current_threshold, max_sleep_ticks, device, hidden_size=hidden_size, episode='Final'
    )

    if args.save_figures:
        fig_path = os.path.join(output_dir, f'money_shot_final_{args.mode}.png')
        generate_money_shot_figure(
            timeline_data, agents, save_path=fig_path, title=f'ETD-MAPPO Final Money Shot ({args.mode.upper()})'
        )

    # Final baseline comparison
    if args.mode == 'etd':
        comparison_summary, _ = run_baseline_comparison(
            env, actors, device, hidden_size=hidden_size, num_eval_episodes=20
        )
        log_comparison_table(comparison_summary, update_num='Final')

    # Final temporal specialization
    log_temporal_specialization(tracker, 'Final')

    print('\n[*] Training Complete!')
    wandb.finish()


if __name__ == '__main__':
    main()
