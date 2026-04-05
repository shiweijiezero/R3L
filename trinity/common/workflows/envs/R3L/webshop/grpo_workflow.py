# -*- coding: utf-8 -*-
import os
from pathlib import Path
from typing import List, Optional

import torch
from jinja2 import Environment, FileSystemLoader

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.envs.R3L.webshop import utils
from trinity.common.workflows.workflow import WORKFLOWS, Task, Workflow


@WORKFLOWS.register_module("grpo_baseline_webshop_workflow")
class GRPOBaselineWebshopWorkflow(Workflow):
    """
    GRPO Baseline Workflow for WebShop environment.
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
        self.max_env_steps = 15
        self.max_tokens = 512
        self.task = task
        self.is_eval = task.is_eval
        self.whether_save_data = False

        # Create data directories
        self.data_dir = f"grpo_baseline_webshop_data"
        self.eval_dir = os.path.join(self.data_dir, "eval")
        self.train_dir = os.path.join(self.data_dir, "train")

        os.makedirs(self.eval_dir, exist_ok=True)
        os.makedirs(self.train_dir, exist_ok=True)

        # Initialize WebShop environment
        try:
            import sys

            # Add WebShop path - can be overridden via WEBSHOP_PATH environment variable
            webshop_path = os.environ.get("WEBSHOP_PATH")
            if webshop_path:
                sys.path.append(webshop_path)
            else:
                # sys.path.append("/path/to/webshop")
                sys.path.append("/path/to/webshop")
            import gym
            from web_agent_site.envs import WebAgentTextEnv  # noqa: F401

            # NOTE: Hosting the env requires ~15GB CPU memory.
            # If you want easier env, you can set the num_products to 1000 or 100000.
            self.env = gym.make(
                "WebAgentTextEnv-v0",
                observation_mode="text_rich",
                num_products=None,
                human_goals=True,
            )
        except Exception as e:
            error_message = (
                f"Error importing WebAgentTextEnv {str(e)}. "
                f"Please make sure you have installed the web_agent_site package, "
                f"following the instructions in https://github.com/princeton-nlp/WebShop"
            )
            raise ImportError(error_message)

        # Initialize Jinja2 templates
        prompts_dir = Path(__file__).parent / "prompts"
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(prompts_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Cache templates to avoid repeated loading
        self.webshop_system_template = self.jinja_env.get_template("webshop_system.j2")

        print(f"Initializing GRPOBaselineWebshopWorkflow, temperature={self.temperature}")
        self.reset(task)

    def reset(self, task: Task):
        """Reset the workflow with a new task"""
        self.session_id = int(task.task_desc or "0")
        self.is_eval = task.is_eval
        self.task = task
        self.n = task.repeat_times
        self.temperature = getattr(task.rollout_args, "temperature", 1.0)

    def run(self) -> List[Experience]:
        """Run the GRPO baseline workflow and return experiences"""

        if self.is_eval:
            return utils.eval_webshop(self)

        # Single rollout execution
        exp_lst = []
        for i in range(self.n):
            try:
                trajectory, reward, done, steps, format_valid = utils.first_rollout(
                    self, self.env, self.session_id
                )
                print(f"[GRPO] First rollout - reward: {reward}, steps: {steps}")
                exp = self.model.convert_messages_to_experience(trajectory[:-1])
                exp.reward = reward
                exp.metrics = {
                    "success": 1.0 if reward >= 1.0 else 0.0,
                    "steps": steps,
                    "reward": reward,
                }
                exp_lst.append(exp)
            except Exception:
                pass

        return exp_lst

    def set_repeat_times(self, repeat_times, run_id_base):
        self.repeat_times = repeat_times
        self.run_id_base = run_id_base
        self.n = repeat_times
