# -*- coding: utf-8 -*-
import os
from pathlib import Path
from typing import List, Optional

import torch
from jinja2 import Environment, FileSystemLoader

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.envs.R3L.dapo import utils
from trinity.common.workflows.workflow import WORKFLOWS, Task, Workflow


@WORKFLOWS.register_module("grpo_baseline_dapo_workflow")
class GRPOBaselineDapoWorkflow(Workflow):
    """
    GRPO Baseline Workflow for DAPO mathematical problem solving.
    Performs simple rollouts without reflection or learning from experience.
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
        self.max_attempts = 3
        self.max_tokens = 4096
        self.max_reflect_tokens = 4096
        self.task = task
        self.is_eval = task.is_eval
        self.whether_save_data = True

        # Create data directories
        self.data_dir = f"grpo_baseline_dapo_data"
        self.eval_dir = os.path.join(self.data_dir, "eval")
        self.train_dir = os.path.join(self.data_dir, "train")

        os.makedirs(self.eval_dir, exist_ok=True)
        os.makedirs(self.train_dir, exist_ok=True)

        # Initialize Jinja2 templates
        prompts_dir = Path(__file__).parent / "prompts"
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(prompts_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Cache templates to avoid repeated loading
        self.dapo_system_template = self.jinja_env.get_template("math_system.j2")

        print(f"Initializing GRPOBaselineDapoWorkflow, temperature={self.temperature}")
        self.reset(task)

    def reset(self, task: Task):
        """Reset the workflow with a new task"""
        self.is_eval = task.is_eval
        self.task = task
        self.n = task.repeat_times
        self.temperature = getattr(task.rollout_args, "temperature", 1.0)

        # Extract prompt and ground truth from task
        if hasattr(task, "raw_task") and task.raw_task:
            raw_task = task.raw_task

            # Format 1: prompt is a list (math_dapo format)
            if "prompt" in raw_task and isinstance(raw_task["prompt"], list):
                if len(raw_task["prompt"]) > 0 and isinstance(raw_task["prompt"][0], dict):
                    self.prompt = raw_task["prompt"][0].get("content", "")
                else:
                    self.prompt = ""

                reward_model_data = raw_task.get("reward_model", {})
                if isinstance(reward_model_data, dict):
                    self.ground_truth = reward_model_data.get("ground_truth", "")
                else:
                    self.ground_truth = ""

            # Format 2: question/answer format (AIME format)
            elif "question" in raw_task and "answer" in raw_task:
                self.prompt = raw_task.get("question", "")
                self.ground_truth = raw_task.get("answer", "")

            # Fallback: simple prompt/answer
            else:
                self.prompt = raw_task.get("prompt", "")
                self.ground_truth = raw_task.get("answer", "")
        else:
            self.prompt = ""
            self.ground_truth = ""

    def run(self) -> List[Experience]:
        """Run the GRPO baseline workflow and return experiences"""

        if self.is_eval:
            return utils.eval_dapo(self)

        # Generate unique task ID
        task_id = f"{str(self.task.batch_id).replace('/', '_')}_{self.task.task_id}"

        # Single rollout execution
        exp_lst = []
        for i in range(self.n):
            try:
                (
                    trajectory,
                    reward,
                    success,
                    predicted_answer,
                    ground_truth,
                    attempts,
                ) = utils.first_rollout(self)
                print(f"[GRPO] First rollout - reward: {reward}, attempts: {attempts}")
                exp = self.model.convert_messages_to_experience(trajectory[:-1])
                exp.reward = reward
                exp.metrics = {
                    "success": 1.0 if success else 0.0,
                    "reward": reward,
                    "attempts": attempts,
                }
                exp_lst.append(exp)

                if self.whether_save_data:
                    # Save training experience data
                    train_record = utils.create_experience_record(
                        task_id=task_id,
                        trajectory=trajectory,
                        reward=reward,
                        success=success,
                        predicted_answer=predicted_answer,
                        ground_truth=ground_truth,
                        attempt_type="train",
                    )
                    utils.save_experience_data(
                        task_id=f"{task_id}_attempt_{i}",
                        experience_data=train_record,
                        data_dir=self.train_dir,
                    )
            except Exception:
                pass

        return exp_lst

    def set_repeat_times(self, repeat_times, run_id_base):
        self.repeat_times = repeat_times
        self.run_id_base = run_id_base
        self.n = repeat_times
