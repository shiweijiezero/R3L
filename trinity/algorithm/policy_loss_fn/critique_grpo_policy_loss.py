"""Critique-GRPO policy loss function.

Implements the shaping function for off-policy samples in Critique-GRPO.
- On-policy samples: standard PPO clipping
- Off-policy samples: f(pi) = pi / (pi + gamma) shaping

Reference: https://arxiv.org/abs/2505.xxxx (Critique-GRPO paper)
"""

from typing import Dict, Optional, Tuple

import torch

from trinity.algorithm.policy_loss_fn.policy_loss_fn import POLICY_LOSS_FN, PolicyLossFn
from trinity.algorithm.utils import masked_loss, masked_mean


@POLICY_LOSS_FN.register_module("critique_grpo")
class CritiqueGRPOPolicyLossFn(PolicyLossFn):
    """Policy loss for Critique-GRPO with on/off-policy handling.

    For on-policy samples (is_off_policy=False):
        Standard PPO ratio with clipping: ratio = exp(log_pi - log_mu)

    For off-policy samples (is_off_policy=True):
        Shaping function: f(pi) = pi / (pi + gamma)
        This ensures bounded gradient even for very low probability tokens.
    """

    def __init__(
        self,
        backend: str = "verl",
        clip_range: Optional[float] = None,
        clip_range_low: Optional[float] = None,
        clip_range_high: Optional[float] = None,
        gamma: float = 0.1,
        loss_agg_mode: Optional[str] = "token-mean",
    ) -> None:
        """Initialize Critique-GRPO policy loss.

        Args:
            backend: Training backend ("verl" or "megatron").
            clip_range: PPO clipping range (used if clip_range_low/high not specified).
            clip_range_low: Lower bound for PPO clipping (default: clip_range).
            clip_range_high: Upper bound for PPO clipping (default: clip_range).
            gamma: Shaping parameter for off-policy samples. Default 0.1.
                   f(pi) = pi / (pi + gamma)
            loss_agg_mode: Loss aggregation mode ("token-mean" or "sample-mean").
        """
        super().__init__(backend=backend)
        if clip_range_low is None:
            self.clip_range_low = clip_range
        else:
            self.clip_range_low = clip_range_low
        if clip_range_high is None:
            self.clip_range_high = clip_range
        else:
            self.clip_range_high = clip_range_high
        assert self.clip_range_low is not None, "clip_range_low must be specified."
        assert self.clip_range_high is not None, "clip_range_high must be specified."
        self.gamma = gamma
        self.loss_agg_mode = loss_agg_mode

    def __call__(
        self,
        logprob: torch.Tensor,
        old_logprob: torch.Tensor,
        action_mask: torch.Tensor,
        advantages: torch.Tensor,
        is_off_policy: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute policy loss with on/off-policy handling.

        Args:
            logprob: Current policy log probabilities. Shape: (batch_size, seq_len)
            old_logprob: Old policy log probabilities. Shape: (batch_size, seq_len)
            action_mask: Action mask. Shape: (batch_size, seq_len)
            advantages: Advantage values. Shape: (batch_size, seq_len)
            is_off_policy: Boolean tensor indicating off-policy samples.
                           Shape: (batch_size,) or (batch_size, 1)

        Returns:
            pg_loss: Policy gradient loss.
            metrics: Dictionary of metrics for logging.
        """
        # Compute PPO ratio for on-policy
        negative_approx_kl = logprob - old_logprob
        ratio = torch.exp(negative_approx_kl)
        ppo_kl = masked_mean(-negative_approx_kl, action_mask)

        # On-policy loss with PPO clipping
        on_pg_losses = -advantages * ratio
        on_pg_losses_clipped = -advantages * torch.clamp(
            ratio, 1.0 - self.clip_range_low, 1.0 + self.clip_range_high
        )
        on_pg_losses = torch.max(on_pg_losses, on_pg_losses_clipped)

        # Off-policy loss with shaping function f(pi) = pi / (pi + gamma)
        prob = torch.exp(logprob)
        off_ratio = prob / (prob + self.gamma)
        off_pg_losses = -advantages * off_ratio

        # Handle case when is_off_policy is not provided (all on-policy)
        if is_off_policy is None:
            pg_loss = masked_loss(on_pg_losses, action_mask, loss_agg_mode=self.loss_agg_mode)
            pg_clipfrac = masked_mean(
                torch.gt(on_pg_losses_clipped, -advantages * ratio).float(), action_mask
            )
            metrics = {
                "pg_clipfrac": pg_clipfrac.detach().item(),
                "ppo_kl": ppo_kl.detach().item(),
                "pg_loss": pg_loss.detach().item(),
                "on_policy_ratio": 1.0,
                "off_policy_ratio": 0.0,
            }
            return pg_loss, metrics

        # Expand is_off_policy to match tensor dimensions
        if is_off_policy.dim() == 1:
            is_off_policy = is_off_policy.unsqueeze(-1)  # (batch_size, 1)

        # Create masks for on/off policy samples
        off_policy_mask = is_off_policy.expand_as(action_mask).float()
        on_policy_mask = 1.0 - off_policy_mask

        # Combined loss: off-policy uses shaping, on-policy uses PPO
        pg_losses = off_pg_losses * off_policy_mask + on_pg_losses * on_policy_mask
        pg_loss = masked_loss(pg_losses, action_mask, loss_agg_mode=self.loss_agg_mode)

        # Compute metrics
        on_policy_action_mask = action_mask * on_policy_mask
        off_policy_action_mask = action_mask * off_policy_mask

        # On-policy clip fraction
        if on_policy_action_mask.sum() > 0:
            on_pg_clipfrac = masked_mean(
                torch.gt(on_pg_losses_clipped, -advantages * ratio).float(),
                on_policy_action_mask,
            )
        else:
            on_pg_clipfrac = torch.tensor(0.0, device=pg_loss.device)

        # Off-policy ratio mean
        if off_policy_action_mask.sum() > 0:
            off_ratio_mean = masked_mean(off_ratio, off_policy_action_mask)
        else:
            off_ratio_mean = torch.tensor(0.0, device=pg_loss.device)

        # Compute on/off loss separately for logging
        if on_policy_action_mask.sum() > 0:
            on_pg_loss = masked_mean(on_pg_losses, on_policy_action_mask)
        else:
            on_pg_loss = torch.tensor(0.0, device=pg_loss.device)

        if off_policy_action_mask.sum() > 0:
            off_pg_loss = masked_mean(off_pg_losses, off_policy_action_mask)
        else:
            off_pg_loss = torch.tensor(0.0, device=pg_loss.device)

        # Count on/off policy samples
        batch_size = is_off_policy.size(0)
        off_count = is_off_policy.sum().item()
        on_count = batch_size - off_count

        metrics = {
            "pg_clipfrac": on_pg_clipfrac.detach().item(),
            "ppo_kl": ppo_kl.detach().item(),
            "pg_loss": pg_loss.detach().item(),
            "on_pg_loss": on_pg_loss.detach().item(),
            "off_pg_loss": off_pg_loss.detach().item(),
            "off_ratio_mean": off_ratio_mean.detach().item(),
            "on_policy_count": on_count,
            "off_policy_count": off_count,
            "on_policy_ratio": on_count / batch_size if batch_size > 0 else 0.0,
            "off_policy_ratio": off_count / batch_size if batch_size > 0 else 0.0,
        }
        return pg_loss, metrics

    @classmethod
    def default_args(cls) -> Dict:
        return {
            "clip_range": 0.2,
            "gamma": 0.1,
            "loss_agg_mode": "token-mean",
        }
