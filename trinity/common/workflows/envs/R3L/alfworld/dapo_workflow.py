# -*- coding: utf-8 -*-
import os
from pathlib import Path
from typing import List, Optional

import torch
from jinja2 import Environment, FileSystemLoader

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.envs.R3L.alfworld import utils
from trinity.common.workflows.workflow import WORKFLOWS, Task, Workflow


@WORKFLOWS.register_module("dapo_alfworld_workflow")
class DAPOAlfworldWorkflow(Workflow):
    """
    DAPO Workflow for Alfworld environment.
    Performs rollouts with DAPO-style overlong penalty on response length.
    No separate reward function needed - penalty computed directly in workflow.
    """

    can_reset: bool = True
    can_repeat: bool = True

    def __init__(
        self,
        model: ModelWrapper,
        task: Task,
        auxiliary_models: Optional[List] = None,
    ):
        super().__init__(
            model=model,
            task=task,
            auxiliary_models=auxiliary_models,
        )
        # Initialize workflow parameters
        self.temperature = getattr(task.rollout_args, "temperature", 1.0)
        self.max_env_steps = 25
        self.max_tokens = 512
        self.task = task
        self.is_eval = task.is_eval
        self.whether_save_data = False

        # Create data directories
        self.data_dir = f"dapo_alfworld_data"
        self.eval_dir = os.path.join(self.data_dir, "eval")
        self.train_dir = os.path.join(self.data_dir, "train")

        os.makedirs(self.eval_dir, exist_ok=True)
        os.makedirs(self.train_dir, exist_ok=True)

        # DAPO overlong penalty parameters
        workflow_args = task.workflow_args or {}
        self.enable_overlong_penalty = workflow_args.get("enable_overlong_penalty", True)
        self.penalty_factor = workflow_args.get("penalty_factor", 1.0)
        self.max_response_length = workflow_args.get("max_response_length", 512)
        self.cache_length = workflow_args.get("cache_length", 100)

        # Initialize Jinja2 templates
        prompts_dir = Path(__file__).parent / "prompts"
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(prompts_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Cache templates to avoid repeated loading
        self.alfworld_system_template = self.jinja_env.get_template("alfworld_system.j2")

        print(
            f"Initializing DAPOAlfworldWorkflow, temperature={self.temperature}, "
            f"overlong_penalty={'enabled' if self.enable_overlong_penalty else 'disabled'}"
        )
        self.reset(task)

    def reset(self, task: Task):
        """Reset the workflow with a new task"""
        self.game_file_path = task.task_desc or task.raw_task.get("game_file", "")
        self.is_eval = task.is_eval
        self.temperature = getattr(task.rollout_args, "temperature", 1.0)
        self.task = task
        self.n = task.repeat_times

        # Update DAPO parameters if provided
        workflow_args = task.workflow_args or {}
        if "enable_overlong_penalty" in workflow_args:
            self.enable_overlong_penalty = workflow_args["enable_overlong_penalty"]
        if "penalty_factor" in workflow_args:
            self.penalty_factor = workflow_args["penalty_factor"]
        if "max_response_length" in workflow_args:
            self.max_response_length = workflow_args["max_response_length"]
        if "cache_length" in workflow_args:
            self.cache_length = workflow_args["cache_length"]

    def compute_overlong_penalty(self, response_tokens: torch.Tensor) -> float:
        """
        Compute DAPO-style overlong penalty based on response token length.

        Args:
            response_tokens: Response tokens (tensor)

        Returns:
            Penalty score (non-positive float)
        """
        if not self.enable_overlong_penalty:
            return 0.0

        response_len = len(response_tokens)
        expected_len = self.max_response_length - self.cache_length

        if response_len < expected_len:
            # No penalty for short responses
            return 0.0
        elif response_len > self.max_response_length:
            # Fixed penalty for excessively long responses
            return -self.penalty_factor
        else:
            # Linear penalty in the transition zone
            return (expected_len - response_len) / self.cache_length * self.penalty_factor

    def run(self) -> List[Experience]:
        """Run the DAPO workflow and return experiences"""

        if self.is_eval:
            return utils.eval_alfworld(self)

        # Single rollout execution
        env = utils.create_alfworld_environment(self.game_file_path)
        exp_lst = []
        for i in range(self.n):
            try:
                trajectory, reward, done, steps, format_valid = utils.first_rollout(self, env)
                print(f"[DAPO] Rollout - reward: {reward}, steps: {steps}")

                # Convert trajectory to experience
                exp = self.model.convert_messages_to_experience(trajectory[:-1])

                # Extract response tokens from experience
                response_tokens = exp.tokens[exp.prompt_length :]

                # Compute DAPO overlong penalty (format score)
                format_score = self.compute_overlong_penalty(response_tokens)

                # Calculate accuracy score
                accuracy_score = 1.0 if reward >= 1.0 else 0.0

                # Total reward = accuracy + format_score
                total_reward = accuracy_score + format_score

                # Update experience reward and metrics
                exp.reward = total_reward
                exp.metrics = {
                    "success": accuracy_score,
                    "steps": steps,
                    "env_reward": reward,
                    "accuracy": accuracy_score,
                    "format_score": format_score,
                    "response_length": len(response_tokens),
                    "total_reward": total_reward,
                }

                # Set experience ID
                exp.eid.task = str(self.task.task_id)
                exp.eid.run = i + self.run_id_base

                exp_lst.append(exp)
            except Exception as e:
                print(f"[DAPO] Rollout failed: {e}")
                pass

        return exp_lst

    def set_repeat_times(self, repeat_times, run_id_base):
        self.repeat_times = repeat_times
        self.run_id_base = run_id_base
        self.n = repeat_times
