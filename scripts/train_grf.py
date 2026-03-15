"""
ETD-MAPPO Training Script for Google Research Football (GRF)

This script implements comprehensive tracking for:
- Win Rate / Task Performance (Goals scored)
- Money Shot Timeline Visualization
- Per-Agent Temporal Specialization
- Baseline Comparison (Vanilla vs Fixed-Skip vs ETD)
"""

import torch
import torch.optim as optim
import wandb
import argparse
import numpy as np
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.rnn_networks import AsynchronousGRUActor, AsynchronousGRUCritic
from envs.grf_wrapper import AsynchronousGRFWrapper
from algorithms.ppo.loss import compute_masked_ppo_loss
from algorithms.ppo.buffer import AsynchronousRolloutBuffer
from utils.wandb_logger import (
    AdaptiveEntropyScheduler, 
    PerformanceTracker,
    log_episode_timeline,
    generate_money_shot_figure,
    log_temporal_specialization,
    run_baseline_comparison,
    log_comparison_table
)

def parse_args():
    parser = argparse.ArgumentParser(description="ETD-MAPPO GRF Comprehensive Evaluator")
    parser.add_argument("--mode", type=str, default="etd", choices=["etd", "vanilla", "fixed_skip"])
    parser.add_argument("--fixed_skip_amount", type=int, default=4)
    parser.add_argument("--scenario_name", type=str, default="academy_3_vs_1_with_keeper")
    parser.add_argument("--num_left_players", type=int, default=3)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_updates", type=int, default=2000)
    parser.add_argument("--eval_interval", type=int, default=200, help="Run evaluation every N updates")
    parser.add_argument("--save_figures", action="store_true", help="Save Money Shot figures locally")
    return parser.parse_args()


def determine_success(total_reward, episode_length):
    """
    Determine if a GRF episode was successful.
    Success = scored a goal (positive reward).
    """
    return total_reward > 0


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing GRF '{args.scenario_name}' on device: {device} | Mode: {args.mode.upper()}")
    
    # Create output directory for figures
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    wandb.init(
        project="ETD-MAPPO-Research", 
        name=f"GRF-{args.scenario_name}-{args.mode}-comprehensive",
        config=vars(args)
    )
    
    try:
        env = AsynchronousGRFWrapper(
            env_name=args.scenario_name, 
            number_of_left_players_agent_controls=args.num_left_players
        )
    except Exception as e:
        print(f"[!] GRF initialization failed: {e}")
        print("[*] Continuing with mock environment for architecture verification...")
        env = AsynchronousGRFWrapper(
            env_name=args.scenario_name,
            number_of_left_players_agent_controls=args.num_left_players
        )
    
    # Hyperparameters
    num_steps = 128 
    num_updates = args.num_updates
    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.2
    ent_coef = 0.01
    vf_coef = 0.5
    hidden_size = 128  # Larger for GRF
    
    max_sleep_ticks = args.fixed_skip_amount if args.mode == "fixed_skip" else 4
    
    # Mode-specific entropy scheduler
    # GRF has 19 actions, so max entropy is ~2.94
    if args.mode == "vanilla":
        entropy_scheduler = AdaptiveEntropyScheduler(
            initial_threshold=-1.0, final_threshold=-1.0, total_updates=num_updates
        )
    elif args.mode == "fixed_skip":
        entropy_scheduler = AdaptiveEntropyScheduler(
            initial_threshold=999.0, final_threshold=999.0, total_updates=num_updates
        )
    else:
        # ETD: Higher thresholds for larger action space
        entropy_scheduler = AdaptiveEntropyScheduler(
            initial_threshold=1.5, final_threshold=2.8, total_updates=num_updates
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
    
    print(f"[*] Networks constructed for {len(agents)} GRF agents.")
    
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
        episode_reward_sum = 0.0
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
                        max_sleep_ticks=max_sleep_ticks
                    )
                    
                    value, new_critic_h = critics[a](global_obs.unsqueeze(0), critic_hidden[a], active_mask_tensor)
                    
                    actor_hidden[a] = new_actor_h
                    critic_hidden[a] = new_critic_h
                    
                    step_values[a] = value.squeeze()
                    step_logprobs[a] = log_prob.squeeze()
                    step_entropies[a] = entropy.item()
                    
                    # Mode-specific sleep logic
                    if args.mode == "vanilla":
                        final_sleep = 0
                    elif args.mode == "fixed_skip":
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
            
            step_reward = sum(rewards_dict.values())
            episode_reward_sum += step_reward
            
            for a in agents:
                mask_val = active_masks.get(a, 0.0)
                reward = rewards_dict.get(a, 0.0)
                done = dones_dict.get(a, False)
                
                if mask_val == 0.0:
                    saved_inferences += 1
                
                buffers[a].insert_step(
                    obs=next_obs[a],
                    action=torch.tensor(actions_dict[a]).to(device),
                    logprob=step_logprobs[a],
                    value=step_values[a],
                    reward=reward,
                    done=torch.tensor(done, dtype=torch.float32).to(device),
                    active_mask=mask_val
                )
                
                next_obs[a] = torch.tensor(next_obs_dict[a], dtype=torch.float32).to(device)
            
            global_states = next_global_state
            
            # Episode termination
            if all(dones_dict.values()):
                success = determine_success(episode_reward_sum, step + 1)
                tracker.end_episode(success=success)
                episodes_completed += 1
                
                # Reset for new episode
                episode_reward_sum = 0.0
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
                        newlogprob, b_logprobs, newvalue, returns, advantages, 
                        b_masks, clip_coef, vf_coef, ent_coef, entropy
                    )
                    
                    optimizers[a].zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(actors[a].parameters()) + list(critics[a].parameters()), 1.0
                    )
                    optimizers[a].step()
        
        # Compute metrics
        flop_reduction_pct = (saved_inferences / total_evaluations) * 100
        perf_metrics = tracker.get_metrics()
        
        # Calculate average entropy for this update
        avg_entropy = np.mean([step_entropies[a] for a in agents]) if step_entropies else 0.0
        
        # Log to WandB - include more metrics
        log_dict = {
            "compute/FLOP_reduction_pct": flop_reduction_pct,
            "compute/saved_inferences": saved_inferences,
            "dynamics/entropy_threshold": current_threshold,
            "dynamics/avg_policy_entropy": avg_entropy,
            "training/episodes_completed": episodes_completed,
            "training/episode_reward_this_update": episode_reward_sum,
            **perf_metrics
        }
        
        # Add per-agent entropy
        for a in agents:
            if a in step_entropies:
                log_dict[f"entropy/{a}"] = step_entropies[a]
        
        wandb.log(log_dict, step=update)
        
        # Periodic detailed logging
        if update % args.eval_interval == 0:
            print(f"\n{'='*70}")
            print(f"Update {update}/{num_updates} | Mode: {args.mode.upper()}")
            print(f"{'='*70}")
            print(f"  FLOP Reduction: {flop_reduction_pct:.1f}%")
            print(f"  Entropy Threshold: {current_threshold:.3f}")
            print(f"  Avg Policy Entropy: {avg_entropy:.3f}")
            
            if "performance/win_rate" in perf_metrics:
                print(f"  Win Rate: {perf_metrics['performance/win_rate']:.1f}%")
            if "performance/episode_reward_mean" in perf_metrics:
                print(f"  Mean Episode Reward: {perf_metrics['performance/episode_reward_mean']:.2f}")
            
            # Log temporal specialization
            log_temporal_specialization(tracker, update)
            
            # Generate Money Shot timeline
            print(f"\n[*] Generating Money Shot Timeline...")
            timeline_data, total_reward = log_episode_timeline(
                env, actors, current_threshold, max_sleep_ticks, 
                device, hidden_size=hidden_size, episode=f"Update{update}"
            )
            
            # Save figure locally if requested
            if args.save_figures:
                fig_path = os.path.join(output_dir, f"grf_money_shot_update_{update}.png")
                generate_money_shot_figure(
                    timeline_data, agents, save_path=fig_path,
                    title=f"GRF Money Shot (Update {update}, {args.mode.upper()})"
                )
            
            # Run baseline comparison
            if update % (args.eval_interval * 2) == 0 and args.mode == "etd":
                print(f"\n[*] Running Baseline Comparison...")
                comparison_summary, _ = run_baseline_comparison(
                    env, actors, device, hidden_size=hidden_size, num_eval_episodes=10
                )
                log_comparison_table(comparison_summary, update)
        
        # Brief progress update
        if update % 50 == 0:
            print(f"--- GRF Update {update}/{num_updates} | FLOP: {flop_reduction_pct:.1f}% | Entropy: {avg_entropy:.3f} | Episodes: {episodes_completed} | Reward: {episode_reward_sum:.3f} ---")
    
    # Final evaluation
    print(f"\n{'='*70}")
    print("FINAL EVALUATION")
    print(f"{'='*70}")
    
    timeline_data, total_reward = log_episode_timeline(
        env, actors, current_threshold, max_sleep_ticks,
        device, hidden_size=hidden_size, episode="Final"
    )
    
    if args.save_figures:
        fig_path = os.path.join(output_dir, f"grf_money_shot_final_{args.mode}.png")
        generate_money_shot_figure(
            timeline_data, agents, save_path=fig_path,
            title=f"GRF Final Money Shot ({args.mode.upper()})"
        )
    
    if args.mode == "etd":
        comparison_summary, _ = run_baseline_comparison(
            env, actors, device, hidden_size=hidden_size, num_eval_episodes=20
        )
        log_comparison_table(comparison_summary, update_num="Final")
    
    log_temporal_specialization(tracker, "Final")
    
    print("\n[*] Training Complete!")
    wandb.finish()


if __name__ == "__main__":
    main()
