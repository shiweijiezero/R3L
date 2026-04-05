"""GRPO advantage computation for multi-step scenarios
"""
from typing import Dict, List, Optional, Tuple

import torch

from trinity.algorithm.advantage_fn.advantage_fn import ADVANTAGE_FN, AdvantageFn
from trinity.buffer.operators import ExperienceOperator
from trinity.common.experience import Experience, group_by
from trinity.utils.monitor import gather_metrics


@ADVANTAGE_FN.register_module("step_wise_grpo")
class StepWiseGRPOAdvantageFn(AdvantageFn, ExperienceOperator):
    """
    An advantage function that broadcasts advantages from the last step to previous steps.
    Inspired by rLLM (https://github.com/rllm-org/rllm).
    """

    def __init__(
        self,
        epsilon: float = 1e-6,
        enable_step_norm: bool = False,
        std_cal_level: str = "group",  # 'group' (task-level) or 'batch'
        std_threshold: Optional[float] = None,
        **kwargs,
    ) -> None:
        """Initialize the Step-wise GRPO advantage function.

        Args:
            epsilon (float): A small value to avoid division by zero.
            enable_step_norm (bool): If True, normalize advantages by trajectory length.
            std_cal_level (str): The scope for calculating reward standard deviation.
                'group' (default): Std is calculated per task group.
                'batch': Std is calculated across all last-step rewards in the entire batch.
                The mean is always calculated per task group.
            std_threshold (Optional[float]): If provided, task groups with a reward standard deviation
                equal or below this threshold will be skipped.
        """
        self.epsilon = epsilon
        self.enable_step_norm = enable_step_norm
        self.std_cal_level = std_cal_level
        self.std_threshold = std_threshold
        if self.std_cal_level not in ["group", "batch"]:
            raise ValueError("std_cal_level must be either 'group' or 'batch'")

    def calculate_last_step_advantage(
        self,
        exps: Dict[str, Experience],
        precomputed_std: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, float], Dict[str, float], bool]:
        """Calculate group advantage for a given group of experiences.

        Args:
            exps (Dict[str, Experience]): One experience per run, keyed by run ID.
            precomputed_std (Optional[torch.Tensor]): Precomputed standard deviation for batch-level calculation.

        Returns:
            Dict[str, float]: Scores for each run.
            Dict[str, float]: Metrics for logging.
            bool: Whether this group should be skipped.
        """
        with torch.no_grad():
            if len(exps) == 1:
                group_reward_mean = torch.tensor(0.0)
                group_reward_std = torch.tensor(1.0)
            else:
                rewards = torch.tensor([exp.reward for exp in exps.values()], dtype=torch.float32)
                group_reward_mean = torch.mean(rewards)
                group_reward_std = torch.std(rewards)

            # Determine if this group should be skipped based on std_threshold
            should_skip = False
            if self.std_threshold is not None:
                if len(exps) == 1 or group_reward_std <= self.std_threshold:
                    should_skip = True

            scores = {}
            for rid, exp in exps.items():
                if self.std_cal_level == "batch" and precomputed_std is not None:
                    score = (exp.reward - group_reward_mean) / (precomputed_std + self.epsilon)
                else:
                    score = (exp.reward - group_reward_mean) / (group_reward_std + self.epsilon)
                scores[rid] = score.item()
            metrics = {
                "reward_mean": group_reward_mean.item(),
                "reward_std": group_reward_std.item(),
            }
        return scores, metrics, should_skip

    def broadcast_advantages(
        self, run_exps: Dict[str, List[Experience]], scores: Dict[str, float]
    ) -> Dict[str, List[Experience]]:
        """Broadcast the calculated advantages to all previous steps in each run.

        Args:
            run_exps (Dict[str, List[Experience]]): Experiences grouped by run ID.
            scores (Dict[str, float]): Calculated scores for each run.

        Returns:
            Dict[str, List[Experience]]: Updated experiences with advantages broadcasted.
        """
        for run_id, exps in run_exps.items():
            score = scores[run_id]
            traj_length = len(exps)
            for exp in exps:
                exp.advantages = exp.action_mask * score  # type: ignore [operator]
                if self.enable_step_norm:
                    exp.advantages /= traj_length
                exp.returns = exp.advantages.clone()
        return run_exps

    def process(self, exps: List[Experience]) -> Tuple[List[Experience], Dict]:
        if len(exps) == 0:
            return [], {}
        cnt = 0
        metric_list = []
        filtered_count = 0
        # Step 1: split the experiences into sub-groups by task
        task_exps = group_by(exps, "task")

        # --- Pre-computation step for batch-level standard deviation ---
        precomputed_std = None
        if self.std_cal_level == "batch":
            all_laststep_rewards = []
            for task_exp in task_exps.values():
                # First, group all experiences by run to find the last step of each run
                task_run_exps = group_by(task_exp, "run")
                # Collect rewards from the last step of every run in the entire batch
                last_step_rewards = [
                    run_steps[-1].reward for run_steps in task_run_exps.values() if run_steps
                ]
                all_laststep_rewards.extend(last_step_rewards)

            if len(all_laststep_rewards) <= 1:
                precomputed_std = torch.tensor(1.0)
            else:
                precomputed_std = torch.std(torch.tensor(all_laststep_rewards, dtype=torch.float32))
        # --- End of pre-computation ---

        # Step 2: further split each task's experiences into sub-groups by run
        result_exps = []
        total_task_groups = len(task_exps)
        skipped_task_groups = 0

        for task_exp in task_exps.values():
            run_exps = group_by(task_exp, "run")

            # Step3: extract the last experience (last step) from each run and calculate scores
            last_step_exps = {run_id: step_exps[-1] for run_id, step_exps in run_exps.items()}
            scores, metrics, should_skip = self.calculate_last_step_advantage(
                last_step_exps, precomputed_std=precomputed_std
            )

            # Skip this task group if std is below threshold
            if should_skip:
                # Count all experiences in this task group as filtered
                task_exp_count = sum(len(step_exps) for step_exps in run_exps.values())
                filtered_count += task_exp_count
                skipped_task_groups += 1
                metric_list.append(metrics)
                continue

            metric_list.append(metrics)

            # Step 4: broadcast the advantages to all previous steps
            run_exps = self.broadcast_advantages(run_exps, scores)
            for exps in run_exps.values():
                cnt += len(exps)
                result_exps.extend(exps)

        metrics = gather_metrics(metric_list, "group_advantages")
        metrics["experience_count"] = cnt
        metrics["filtered_count"] = filtered_count

        # Calculate the ratio of skipped task groups
        if total_task_groups > 0:
            metrics["skipped_group_ratio"] = skipped_task_groups / total_task_groups
        else:
            metrics["skipped_group_ratio"] = 0.0

        return result_exps, metrics

    def __call__(self, exps, **kwargs):
        return self.process(exps)

    @classmethod
    def compute_in_trainer(cls) -> bool:
        """Whether the advantage should be computed in the trainer loop."""
        return False

    @classmethod
    def default_args(cls) -> Dict:
        """Return the default configuration for this strategy."""
        return {
            "epsilon": 1e-6,
            "enable_step_norm": False,
            "std_threshold": None,
            "std_cal_level": "group",
        }


@ADVANTAGE_FN.register_module("step_wise_opmd")
class StepWiseOPMDAdvantageFn(StepWiseGRPOAdvantageFn):
    """Step-wise OPMD advantage function.

    Same step-level structure as StepWiseGRPOAdvantageFn but uses OPMD baseline
    (mean subtraction without std normalization).
    """

    def calculate_last_step_advantage(
        self,
        exps: Dict[str, Experience],
        precomputed_std=None,
    ) -> Tuple[Dict[str, float], Dict[str, float], bool]:
        with torch.no_grad():
            if len(exps) == 1:
                group_reward_mean = torch.tensor(0.0)
            else:
                rewards = torch.tensor([exp.reward for exp in exps.values()], dtype=torch.float32)
                group_reward_mean = torch.mean(rewards)

            scores = {}
            for rid, exp in exps.items():
                scores[rid] = (exp.reward - group_reward_mean.item())
            metrics = {
                "reward_mean": group_reward_mean.item(),
            }
        return scores, metrics, False

    @classmethod
    def default_args(cls) -> Dict:
        return {
            "enable_step_norm": False,
        }


@ADVANTAGE_FN.register_module("step_wise_opmd_reweight_adv")
class StepWiseOPMDReweightAdvAdvantageFn(StepWiseGRPOAdvantageFn):
    """Step-wise OPMD advantage with R3L's positive amplification reweighting.

    Same step-level structure as StepWiseGRPOAdvantageFn (group by task, then run,
    use last-step reward, broadcast to all steps), but with OPMD baseline and
    reweighting rules:
    - Baseline: group mean (OPMD style, no std normalization)
    - If reward >= 1.0 (max reward), set score = 1.0
    - If score >= 0, multiply by alpha (default 3.0) for positive amplification
    - If score < 0, keep unchanged
    """

    def __init__(
        self,
        opmd_baseline: str = "mean",
        tau: float = 1.0,
        alpha: float = 3.0,
        enable_step_norm: bool = False,
        **kwargs,
    ) -> None:
        # Don't call super().__init__() with GRPO-specific args
        self.opmd_baseline = opmd_baseline
        self.tau = tau
        self.alpha = alpha
        self.enable_step_norm = enable_step_norm

    def calculate_last_step_advantage(
        self,
        exps: Dict[str, Experience],
        precomputed_std=None,
    ) -> tuple:
        """Calculate OPMD reweighted advantage from last-step rewards."""
        with torch.no_grad():
            if len(exps) == 1:
                group_baseline = torch.tensor(0.0)
                group_rewards = torch.tensor(
                    [list(exps.values())[0].reward], dtype=torch.float32
                )
            else:
                group_rewards = torch.tensor(
                    [exp.reward for exp in exps.values()], dtype=torch.float32
                )
                if self.opmd_baseline == "mean":
                    group_baseline = torch.mean(group_rewards)
                else:
                    group_baseline = self.tau * (
                        torch.logsumexp(group_rewards / self.tau, dim=-1)
                        - torch.log(torch.tensor(len(exps)))
                    )

            scores = {}
            for rid, exp in exps.items():
                score = exp.reward - group_baseline.item()
                # R3L reweighting: positive amplification
                if exp.reward >= 1.0:
                    score = 1.0
                elif score >= 0:
                    score = score * self.alpha
                # Negative scores unchanged
                scores[rid] = score

            metrics = {
                "group_baseline": group_baseline.item(),
                "reward_mean": torch.mean(group_rewards).item(),
            }
        # Never skip for OPMD reweight (no std threshold)
        return scores, metrics, False

    def process(self, exps: List[Experience]) -> tuple:
        if len(exps) == 0:
            return [], {}
        cnt = 0
        metric_list = []

        task_exps = group_by(exps, "task")
        result_exps = []

        for task_exp in task_exps.values():
            run_exps = group_by(task_exp, "run")
            last_step_exps = {
                run_id: step_exps[-1] for run_id, step_exps in run_exps.items()
            }
            scores, metrics, _ = self.calculate_last_step_advantage(last_step_exps)
            metric_list.append(metrics)

            run_exps = self.broadcast_advantages(run_exps, scores)
            for step_exps in run_exps.values():
                cnt += len(step_exps)
                result_exps.extend(step_exps)

        metrics = gather_metrics(metric_list, "group_advantages")
        metrics["experience_count"] = cnt
        metrics["filtered_count"] = 0
        return result_exps, metrics

    @classmethod
    def default_args(cls) -> Dict:
        return {
            "opmd_baseline": "mean",
            "tau": 1.0,
            "alpha": 3.0,
            "enable_step_norm": False,
        }
