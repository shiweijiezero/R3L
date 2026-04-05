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


@WORKFLOWS.register_module("reflect_grpo_webshop_workflow")
class ReflectGRPOWebshopWorkflow(Workflow):
    """
    Reflect GRPO Workflow for WebShop environment.
    Uses standard GRPO algorithm with half rollout and half reflection+retry.
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
        self.max_reflect_tokens = 4096
        self.task = task
        self.is_eval = task.is_eval
        self.whether_save_data = False

        # Create data directories
        self.data_dir = f"reflect_grpo_webshop_data"
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
        self.reflection_template = self.jinja_env.get_template("reflection.j2")

        print(f"Initializing ReflectGRPOWebshopWorkflow, temperature={self.temperature}")
        self.reset(task)

    def reset(self, task: Task):
        """Reset the workflow with a new task"""
        self.session_id = int(task.task_desc or "0")
        self.is_eval = task.is_eval
        self.task = task
        self.n = task.repeat_times
        self.temperature = getattr(task.rollout_args, "temperature", 1.0)

    def get_reflect(self, trajectory):
        """Generate reflection on a failed trajectory"""
        formatted_trajectory = utils.format_trajectory_for_reflection(trajectory)
        reflect_prompt = self.reflection_template.render()

        try:
            responses = self.model.chat(
                [
                    {"role": "system", "content": reflect_prompt},
                    {
                        "role": "user",
                        "content": "Here is last attempt trajectory log: \n\n"
                        + formatted_trajectory
                        + "\n\nPlease output in the specified JSON format.",
                    },
                ],
                n=1,
                temperature=self.temperature,
                max_tokens=self.max_reflect_tokens,
            )
            reflection_text = responses[0].response_text.strip()

            # Find first '{' and last '}'
            first_brace = reflection_text.find("{")
            last_brace = reflection_text.rfind("}")

            if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
                json_str = reflection_text[first_brace : last_brace + 1]
            else:
                json_str = reflection_text

            import json

            reflection_data = json.loads(json_str)
            return reflection_data, reflection_text

        except Exception as e:
            print(f"[ReflectGRPO] Reflection failed - Error: {str(e)}")
            return None, None

    def run(self) -> List[Experience]:
        """Run the Reflect GRPO workflow and return experiences"""

        if self.is_eval:
            return utils.eval_webshop(self)

        exp_lst = []

        # First half: normal rollout
        for i in range(self.n // 2):
            try:
                trajectory, reward, done, steps, format_valid = utils.first_rollout(
                    self, self.env, self.session_id
                )
                print(f"[ReflectGRPO] Rollout {i} - reward: {reward}, steps: {steps}")
                exp = self.model.convert_messages_to_experience(trajectory[:-1])
                exp.reward = reward
                exp.metrics = {
                    "success": 1.0 if reward >= 1.0 else 0.0,
                    "steps": steps,
                    "reward": reward,
                }
                exp_lst.append(exp)
            except Exception as e:
                print(f"[ReflectGRPO] Rollout {i} failed - Error: {str(e)}")

        # Second half: reflection + retry
        for i in range(self.n // 2, self.n):
            try:
                # First attempt
                trajectory, reward, done, steps, format_valid = utils.first_rollout(
                    self, self.env, self.session_id
                )
                print(f"[ReflectGRPO] First attempt {i} - reward: {reward}, steps: {steps}")

                exp = self.model.convert_messages_to_experience(trajectory[:-1])
                exp.reward = reward
                exp.metrics = {
                    "success": 1.0 if reward >= 1.0 else 0.0,
                    "steps": steps,
                    "reward": reward,
                }
                exp_lst.append(exp)

                # If not successful, do reflection and retry from the beginning
                if reward < 1.0:
                    print(f"[ReflectGRPO] Failed attempt {i}, attempting reflection and retry from beginning...")
                    reflect_checklist, reflection_text = self.get_reflect(trajectory)
                    is_valid, is_perfect = utils.validate_reflect_report(reflect_checklist, steps)

                    if is_valid and not is_perfect:
                        print(f"[ReflectGRPO] Valid reflection generated, retrying from the beginning...")
                        guidance_prompt = utils.reflect_report_to_guidance_prompt(reflect_checklist)
                        # Always retry from the beginning (retry_step = 0)
                        retry_step = 0

                        try:
                            (
                                distill_trajectory,
                                second_trajectory,
                                second_reward,
                                second_done,
                                second_steps,
                                second_format_valid,
                            ) = utils.second_rollout(self, guidance_prompt, trajectory, retry_step)

                            print(
                                f"[ReflectGRPO] Retry {i} - reward: {second_reward}, steps: {second_steps}, improved: {second_reward > reward}"
                            )

                            second_exp = self.model.convert_messages_to_experience(
                                distill_trajectory[:-1]
                            )
                            second_exp.reward = second_reward
                            second_exp.metrics = {
                                "success": 1.0 if second_reward >= 1.0 else 0.0,
                                "steps": second_steps,
                                "reward": second_reward,
                                "improved": 1.0 if second_reward > reward else 0.0,
                            }
                            exp_lst.append(second_exp)

                        except Exception as e:
                            print(f"[ReflectGRPO] Retry {i} failed - Error: {str(e)}")
                    else:
                        if is_perfect:
                            print(
                                f"[ReflectGRPO] Reflection indicates perfect attempt - No retry needed"
                            )
                        else:
                            print(f"[ReflectGRPO] Invalid reflection - Skipping retry")

            except Exception as e:
                print(f"[ReflectGRPO] Attempt {i} failed - Error: {str(e)}")

        return exp_lst

    def set_repeat_times(self, repeat_times, run_id_base):
        self.repeat_times = repeat_times
        self.run_id_base = run_id_base
        self.n = repeat_times
