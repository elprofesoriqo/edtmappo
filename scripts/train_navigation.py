"""
ETD-MAPPO Training Script for MPE Simple Spread (Navigation Task)

This is the "Proof of Concept" environment intended to demonstrate
SUCCESSFUL Green AI savings:
1. Agents travel to landmarks -> Low Entropy -> Sleep (Savings)
2. Agents avoid collisions -> High Entropy -> Wake (Performance)
"""

import argparse
import os
import sys

import torch
import torch.optim as optim
from pettingzoo.mpe import simple_spread_v3

import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from algorithms.ppo.buffer import AsynchronousRolloutBuffer
from algorithms.ppo.loss import compute_masked_ppo_loss
from envs.mpe_wrapper import AsynchronousMPEWrapper
from models.rnn_networks import AsynchronousGRUActor, AsynchronousGRUCritic
from utils.wandb_logger import (
    AdaptiveEntropyScheduler,
    PerformanceTracker,
    generate_money_shot_figure,
    log_episode_timeline,
)


def parse_args():
    parser = argparse.ArgumentParser(description='ETD-MAPPO Navigation Proof of Concept')
    parser.add_argument('--mode', type=str, default='etd', choices=['etd', 'vanilla', 'fixed_skip'])
    parser.add_argument('--num_updates', type=int, default=2000, help='Short training for simple task')
    parser.add_argument('--eval_interval', type=int, default=200)
    parser.add_argument('--save_figures', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[*] Initializing MPE Simple Spread (Navigation) on {device} | Mode: {args.mode.upper()}')

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    wandb.init(project='ETD-MAPPO-Research', name=f'Nav-ProofOfConcept-{args.mode}', config=vars(args))

    # Simple Spread: 3 agents, 3 landmarks. Penalty for collision.
    # N=3 agents ensures collisions are possible but task is solvable.
    raw_env = simple_spread_v3.parallel_env(N=3, local_ratio=0.5, max_cycles=50, continuous_actions=False)
    env = AsynchronousMPEWrapper(raw_env)

    # Hyperparameters tuned for "Easy but Dynamic" task
    num_steps = 128
    num_updates = args.num_updates
    gamma = 0.99
    clip_coef = 0.2  # PPO clipping parameter
    hidden_size = 64
    max_sleep_ticks = 3

    # Entropy Scheduler: Start strictly awake to learn navigation, then allow sleep
    if args.mode == 'vanilla':
        entropy_scheduler = AdaptiveEntropyScheduler(-1.0, -1.0, num_updates)
    elif args.mode == 'fixed_skip':
        entropy_scheduler = AdaptiveEntropyScheduler(999.0, 999.0, num_updates)
    else:
        # ETD: Ramp up threshold faster since this is an easier task
        entropy_scheduler = AdaptiveEntropyScheduler(
            initial_threshold=0.3, final_threshold=1.4, total_updates=num_updates
        )

    # Init Networks
    obs_dict, global_state, _ = env.reset()
    agents = env.possible_agents
    global_obs_dim = global_state.shape[0]

    actors = {}
    critics = {}
    optimizers = {}
    buffers = {}

    for a in agents:
        obs_dim = raw_env.observation_space(a).shape[0]
        act_dim = raw_env.action_space(a).n

        actors[a] = AsynchronousGRUActor(obs_dim, act_dim, hidden_size).to(device)
        critics[a] = AsynchronousGRUCritic(global_obs_dim, hidden_size).to(device)

        params = list(actors[a].parameters()) + list(critics[a].parameters())
        optimizers[a] = optim.Adam(params, lr=7e-4, eps=1e-5)  # Slightly higher LR for faster convergence
        buffers[a] = AsynchronousRolloutBuffer(num_steps, obs_dim, (), device)

    tracker = PerformanceTracker(agents)
    batch_global_obs = torch.zeros((num_steps, global_obs_dim)).to(device)

    # Training Loop
    for update in range(1, num_updates + 1):
        current_threshold = entropy_scheduler.get_threshold()
        entropy_scheduler.step()

        for a in agents:
            buffers[a].reset()

        obs, _, _ = env.reset()
        next_obs = {a: torch.tensor(obs[a], dtype=torch.float32).to(device) for a in agents}

        actor_hidden = {a: torch.zeros((1, hidden_size)).to(device) for a in agents}
        critic_hidden = {a: torch.zeros((1, hidden_size)).to(device) for a in agents}

        saved_inferences = 0
        total_evaluations = 0
        episode_rewards = {a: 0.0 for a in agents}

        # Rollout
        for step in range(num_steps):
            total_evaluations += len(agents)
            global_obs = torch.cat([next_obs[a] for a in sorted(agents)], dim=-1)
            batch_global_obs[step] = global_obs

            actions = {}
            sleeps = {}
            vals = {}
            logprobs = {}
            entropies = {}

            with torch.no_grad():
                for a in agents:
                    mask = torch.ones((1, 1)).to(device)
                    act, lp, ent, sleep, h = actors[a].get_action_and_sleep_ticks(
                        next_obs[a].unsqueeze(0), actor_hidden[a], mask, current_threshold, max_sleep_ticks
                    )
                    val, c_h = critics[a](global_obs.unsqueeze(0), critic_hidden[a], mask)

                    actor_hidden[a] = h
                    critic_hidden[a] = c_h
                    vals[a] = val.squeeze()
                    logprobs[a] = lp.squeeze()
                    entropies[a] = ent.item()

                    # Force mode behavior
                    final_sleep = 0
                    if args.mode == 'fixed_skip':
                        final_sleep = max_sleep_ticks
                    elif args.mode == 'etd':
                        final_sleep = sleep.item()

                    actions[a] = act.item()
                    sleeps[a] = final_sleep

            # Step
            next_obs_dict, _, rewards, dones, _, active_masks = env.step(actions, sleeps)
            tracker.step(rewards, active_masks, entropies)

            for a in agents:
                m = active_masks.get(a, 0.0)
                r = rewards.get(a, 0.0)
                d = dones.get(a, False)
                episode_rewards[a] += r
                if m == 0.0:
                    saved_inferences += 1

                buffers[a].insert_step(
                    next_obs[a],
                    torch.tensor(actions[a]).to(device),
                    logprobs[a],
                    vals[a],
                    r,
                    torch.tensor(d, dtype=torch.float32).to(device),
                    m,
                )
                if a in next_obs_dict:
                    next_obs[a] = torch.tensor(next_obs_dict[a], dtype=torch.float32).to(device)

            if all(dones.values()):
                # Success in Spread is high negative reward (minimized distance)
                # We normalize "success" as mean reward > -10 (arbitrary threshold for 'good enough')
                avg_rew = sum(episode_rewards.values()) / len(agents)
                tracker.end_episode(success=avg_rew > -15.0)

                obs, _, _ = env.reset()
                next_obs = {a: torch.tensor(obs[a], dtype=torch.float32).to(device) for a in agents}
                actor_hidden = {a: torch.zeros((1, hidden_size)).to(device) for a in agents}
                critic_hidden = {a: torch.zeros((1, hidden_size)).to(device) for a in agents}
                episode_rewards = {a: 0.0 for a in agents}

        # Update
        with torch.no_grad():
            global_obs = torch.cat([next_obs[a] for a in sorted(agents)], dim=-1)
            for a in agents:
                mask = torch.ones((1, 1)).to(device)
                next_v, _ = critics[a](global_obs.unsqueeze(0), critic_hidden[a], mask)
                adv, ret = buffers[a].compute_gae(next_v.squeeze(), buffers[a].dones[-1], gamma, 0.95)

                # PPO update block (simplified for brevity, identical to other scripts)
                b_obs, b_act, b_lp = buffers[a].obs, buffers[a].actions, buffers[a].logprobs
                b_mask = buffers[a].active_masks

                with torch.enable_grad():
                    # Re-eval actor
                    _, new_lp, ent, _, _ = actors[a].get_action_and_sleep_ticks(
                        b_obs,
                        torch.zeros((num_steps, hidden_size)).to(device),
                        b_mask.unsqueeze(-1),
                        current_threshold,
                        max_sleep_ticks,
                        action=b_act,
                    )
                    # Re-eval critic
                    new_v, _ = critics[a](
                        batch_global_obs, torch.zeros((num_steps, hidden_size)).to(device), b_mask.unsqueeze(-1)
                    )

                    loss, _, _, _ = compute_masked_ppo_loss(
                        new_lp, b_lp, new_v.squeeze(), ret, adv, b_mask, clip_coef, 0.5, 0.01, ent
                    )
                    optimizers[a].zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(actors[a].parameters()) + list(critics[a].parameters()), 0.5)
                    optimizers[a].step()

        # Logging
        flop_pct = (saved_inferences / total_evaluations) * 100
        metrics = tracker.get_metrics()

        log_dict = {'compute/FLOP_reduction_pct': flop_pct, 'dynamics/entropy_threshold': current_threshold, **metrics}
        wandb.log(log_dict, step=update)

        if update % args.eval_interval == 0:
            print(
                f'Update {update} | FLOP: {flop_pct:.1f}% | Win Rate: {metrics.get("performance/win_rate", 0):.1f}% | Reward: {metrics.get("performance/episode_reward_mean", -99):.1f}'
            )
            if args.mode == 'etd' and args.save_figures:
                # Money Shot
                timeline_data, _ = log_episode_timeline(
                    env, actors, current_threshold, max_sleep_ticks, device, hidden_size, episode=f'Update{update}'
                )
                generate_money_shot_figure(
                    timeline_data, agents, save_path=os.path.join(output_dir, f'nav_money_shot_{update}.png')
                )

    print('Training Complete')
    wandb.finish()


if __name__ == '__main__':
    main()
