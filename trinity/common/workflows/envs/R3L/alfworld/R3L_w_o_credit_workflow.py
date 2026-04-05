# -*- coding: utf-8 -*-
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from jinja2 import Environment, FileSystemLoader

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.envs.R3L.alfworld import utils
from trinity.common.workflows.workflow import WORKFLOWS, Task, Workflow


@WORKFLOWS.register_module("R3L_alfworld_w_o_credit_workflow")
class R3LAlfworldWoCreditWorkflow(Workflow):
    """
    R3L workflow for alfworld without credit assignment (ablation study)
    This version removes the credit assignment mechanism (_adjust_action_mask_for_retry)
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
        self.max_reflect_tokens = 4096
        self.task = task
        self.is_eval = task.is_eval

        self.whether_save_data = False
        # Create data directories
        self.data_dir = f"R3L_alfworld_data"
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
        self.alfworld_system_template = self.jinja_env.get_template("alfworld_system.j2")
        self.reflection_template = self.jinja_env.get_template("reflection.j2")

        self.default_exp = Experience(
            tokens=torch.tensor([0, 0], dtype=torch.long),
            prompt_length=1,
            action_mask=torch.tensor([False], dtype=torch.bool),
            logprobs=torch.tensor([0.0], dtype=torch.float),
            metrics={
                "success": 0.0,
                "reward": 0.0,
            },
            reward=0.0,
        )

        self.default_second_exp = Experience(
            tokens=torch.tensor([0, 0], dtype=torch.long),
            prompt_length=1,
            action_mask=torch.tensor([False], dtype=torch.bool),
            logprobs=torch.tensor([0.0], dtype=torch.float),
            metrics={
                "second_success": 0.0,
                "second_reward": 0.0,
            },
            reward=0.0,
        )

        print(
            f"Initializing R3LAlfworldWoCreditWorkflow (w/o credit assignment), temperature={self.temperature}"
        )
        self.reset(task)

    def reset(self, task: Task):
        """Reset the workflow with a new task"""
        self.game_file_path = task.task_desc or task.raw_task.get("game_file", "")
        self.is_eval = task.is_eval
        self.temperature = getattr(task.rollout_args, "temperature", 1.0)
        self.task = task
        self.n = task.repeat_times

    def get_reflect(
        self, trajectory: List[Dict[str, str]]
    ) -> tuple[Optional[Dict[str, Any]], Optional[str], Optional[Any]]:
        """
        Generates a comprehensive reflection report using a single, unified self-interrogation prompt.
        The model first assesses its own performance and then follows the appropriate reflection path.
        """
        # Format trajectory for LLM reading
        formatted_trajectory = utils.format_trajectory_for_reflection(trajectory)

        # Use Jinja2 template to render reflection prompt
        reflect_prompt = self.reflection_template.render()

        # Call model and parse results
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

            reflection_data = json.loads(json_str)
            return reflection_data, reflection_text, responses[0]

        except Exception as e:
            return None, None, None

    def run(self) -> List[Experience]:
        """Run the R3L alfworld workflow and return experiences (without credit assignment)"""

        if self.is_eval:
            return utils.eval_alfworld(self)

        # Generate unique task ID
        task_id = f"{str(self.task.batch_id).replace('/', '_')}_{self.task.task_id}"

        env = utils.create_alfworld_environment(self.game_file_path)
        exp_lst = []
        for i in range(self.n // 2):  # Half for rollout, half for reflection + retry
            try:
                trajectory, reward, done, steps, format_valid = utils.first_rollout(self, env)
                print(f"[R3L w/o credit] First rollout - reward: {reward}, steps: {steps}")
                exp = self.model.convert_messages_to_experience(trajectory[:-1])
                exp.reward = reward
                exp.metrics = {
                    "success": 1.0 if reward >= 1.0 else 0.0,
                    "steps": steps,
                    "reward": reward,
                }
                # Set eid
                exp.eid.task = str(self.task.task_id) + f"_explore"
                exp_run_id = len(exp_lst) + self.run_id_base
                exp.eid.run = exp_run_id
                exp_lst.append(exp)

                if self.whether_save_data:
                    # Save first attempt experience data
                    first_record = utils.create_experience_record(
                        task_id=task_id,
                        trajectory=trajectory,
                        reward=reward,
                        steps=steps,
                        success=reward >= 1.0,
                        attempt_type="first",
                    )
                    utils.save_experience_data(
                        task_id=f"{task_id}_attempt_{i}_first",
                        experience_data=first_record,
                        data_dir=self.train_dir,
                    )

                # Reflect on first attempt
                reflect_checklist, reflection_text, reflect_exp = self.get_reflect(trajectory)
                is_valid, is_perfect = utils.validate_reflect_report(reflect_checklist, steps)

                if not is_valid or is_perfect:
                    # If first attempt reward is 1.0 and reflection gives perfect, record reflection exp
                    if reward >= 1.0 and is_perfect and reflect_exp is not None:
                        reflect_exp.reward = 1.0
                        # Set eid
                        reflect_exp.eid.task = str(self.task.task_id) + f"_reflect_{i}"
                        reflect_exp.eid.run = len(exp_lst) + self.run_id_base
                        exp_lst.append(reflect_exp)

                else:
                    guidance_prompt = utils.reflect_report_to_guidance_prompt(reflect_checklist)
                    # Extract retry_step from validated reflection report (top-level field in alfworld schema)
                    retry_step = reflect_checklist.get("retry_from_step", 0)
                    print(f"[R3L w/o credit] Pivot point (retry_from_step): {retry_step}")

                    try:
                        second_env = utils.create_alfworld_environment(self.game_file_path)
                        (
                            distill_trajectory,
                            second_trajectory,
                            second_reward,
                            second_done,
                            second_steps,
                            second_format_valid,
                        ) = utils.second_rollout(
                            self, second_env, guidance_prompt, trajectory, retry_step
                        )
                        print(
                            f"[R3L w/o credit] Second rollout - reward: {second_reward}, steps: {second_steps}, improve: {second_reward > reward}"
                        )
                        second_exp = self.model.convert_messages_to_experience(
                            distill_trajectory[:-1]
                        )

                        # NOTE: Credit assignment removed for ablation study
                        # No adjustment to action_mask based on retry_step

                        second_exp.reward = second_reward
                        second_exp.metrics = {
                            "second_success": 1.0 if second_reward >= 1.0 else 0.0,
                            "second_steps": second_steps,
                            "second_reward": second_reward,
                            "second_improve": 1.0 if second_reward > reward else 0.0,
                            "second_reward_diff": second_reward - reward,
                            "pivot_point": retry_step,
                        }
                        # Set eid
                        second_exp.eid.task = str(self.task.task_id) + f"_explore"
                        second_exp.eid.run = len(exp_lst) + self.run_id_base
                        exp_lst.append(second_exp)

                        if self.whether_save_data:
                            # Save second attempt experience data
                            second_record = utils.create_experience_record(
                                task_id=task_id,
                                trajectory=second_trajectory,
                                reward=second_reward,
                                steps=second_steps,
                                success=second_reward >= 1.0,
                                attempt_type="second",
                                additional_metrics={
                                    "first_reward": reward,
                                    "improvement": second_reward > reward,
                                    "reward_difference": second_reward - reward,
                                    "step_difference": second_steps - steps,
                                },
                            )
                            utils.save_experience_data(
                                task_id=f"{task_id}_attempt_{i}_second",
                                experience_data=second_record,
                                data_dir=self.train_dir,
                            )

                        # If second attempt score is higher than first, or second is perfect with fewer steps,
                        # record reflection and retry data
                        if (second_reward > reward and second_reward >= 1.0) or (
                            second_reward >= 1.0 and second_steps < steps
                        ):
                            reflect_exp.reward = 1.0
                            # Set eid
                            reflect_exp.eid.task = str(self.task.task_id) + f"_reflect_{i}"
                            reflect_exp.eid.run = len(exp_lst) + self.run_id_base
                            exp_lst.append(reflect_exp)

                            # Convert retry data to exp
                            retry_exp = self.model.convert_messages_to_experience(
                                second_trajectory[:-1]
                            )

                            # NOTE: Credit assignment removed for ablation study
                            # No adjustment to action_mask based on retry_step

                            retry_exp.reward = 1.0
                            # Set eid
                            retry_exp.eid.task = str(self.task.task_id) + f"_retry_{i}"
                            retry_exp.eid.run = len(exp_lst) + self.run_id_base
                            exp_lst.append(retry_exp)

                            print("Reflection and retry led to improvement, recording both...")
                    except Exception as e:
                        print(f"Second rollout after reflection failed: {e}")
            except Exception as e:
                print(f"First rollout failed: {e}")

        return exp_lst

    def set_repeat_times(self, repeat_times, run_id_base):
        self.repeat_times = repeat_times
        self.run_id_base = run_id_base
        self.n = repeat_times
