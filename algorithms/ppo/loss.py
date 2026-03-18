import torch


def compute_masked_ppo_loss(
    logprobs, old_logprobs, values1, values2, returns, advantages, active_masks, clip_coef, vf_coef, ent_coef, entropy
):
    """
    Compute PPO loss with Asynchronous Gradient Masking.

    Ensures that gradients are only propagated for active timesteps.
    Dormant agents (active_mask=0) do not contribute to the loss, preventing
    policy updates based on stale or repeated actions.
    """
    # Policy Loss (Actor)
    ratio = torch.exp(logprobs - old_logprobs)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)

    pg_loss = torch.max(pg_loss1, pg_loss2)

    # Asynchronous Gradient Masking
    pg_loss = pg_loss * active_masks

    # Value Loss (Twin Critics)
    v_loss1 = 0.5 * ((values1 - returns) ** 2)
    v_loss1 = v_loss1 * active_masks

    v_loss2 = 0.5 * ((values2 - returns) ** 2)
    v_loss2 = v_loss2 * active_masks

    v_loss = v_loss1 + v_loss2

    # Entropy Masking
    entropy_masked = entropy * active_masks

    # Safe Mean Calculation
    mask_sum = active_masks.sum() + 1e-8

    pg_loss_mean = pg_loss.sum() / mask_sum
    v_loss_mean = v_loss.sum() / mask_sum
    entropy_mean = entropy_masked.sum() / mask_sum

    # Combined Loss
    loss = pg_loss_mean - ent_coef * entropy_mean + v_loss_mean * vf_coef

    return loss, pg_loss_mean, v_loss_mean, entropy_mean
