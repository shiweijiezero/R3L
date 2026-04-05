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


@WORKFLOWS.register_module("reflect_grpo_dapo_workflow")
class ReflectGRPODapoWorkflow(Workflow):
    """
    Reflect GRPO Workflow for DAPO mathematical problem solving.
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
        self.max_attempts = 3
        self.max_tokens = 4096
        self.max_reflect_tokens = 4096
        self.task = task
        self.is_eval = task.is_eval
        self.whether_save_data = True

        # Create data directories
        self.data_dir = f"reflect_grpo_dapo_data"
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
        self.reflection_template = self.jinja_env.get_template("reflection.j2")

        print(f"Initializing ReflectGRPODapoWorkflow, temperature={self.temperature}")
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
            return utils.eval_dapo(self)

        # Generate unique task ID
        task_id = f"{str(self.task.batch_id).replace('/', '_')}_{self.task.task_id}"

        exp_lst = []

        # First half: normal rollout
        for i in range(self.n // 2):
            try:
                (
                    trajectory,
                    reward,
                    success,
                    predicted_answer,
                    ground_truth,
                    attempts,
                ) = utils.first_rollout(self)
                print(f"[ReflectGRPO] Rollout {i} - reward: {reward}, attempts: {attempts}")
                exp = self.model.convert_messages_to_experience(trajectory[:-1])
                exp.reward = reward
                exp.metrics = {
                    "success": 1.0 if success else 0.0,
                    "reward": reward,
                    "attempts": attempts,
                }
                exp_lst.append(exp)

                if self.whether_save_data:
                    # Save first half training data
                    train_record = utils.create_experience_record(
                        task_id=task_id,
                        trajectory=trajectory,
                        reward=reward,
                        success=success,
                        predicted_answer=predicted_answer,
                        ground_truth=ground_truth,
                        attempt_type="train_first_half",
                    )
                    utils.save_experience_data(
                        task_id=f"{task_id}_attempt_{i}_first_half",
                        experience_data=train_record,
                        data_dir=self.train_dir,
                    )
            except Exception as e:
                print(f"[ReflectGRPO] Rollout {i} failed - Error: {str(e)}")

        # Second half: reflection + retry
        for i in range(self.n // 2, self.n):
            try:
                # First attempt
                (
                    trajectory,
                    reward,
                    success,
                    predicted_answer,
                    ground_truth,
                    attempts,
                ) = utils.first_rollout(self)
                print(f"[ReflectGRPO] First attempt {i} - reward: {reward}, attempts: {attempts}")

                exp = self.model.convert_messages_to_experience(trajectory[:-1])
                exp.reward = reward
                exp.metrics = {
                    "success": 1.0 if success else 0.0,
                    "reward": reward,
                    "attempts": attempts,
                }
                exp_lst.append(exp)

                if self.whether_save_data:
                    # Save first attempt data
                    first_record = utils.create_experience_record(
                        task_id=task_id,
                        trajectory=trajectory,
                        reward=reward,
                        success=success,
                        predicted_answer=predicted_answer,
                        ground_truth=ground_truth,
                        attempt_type="first",
                    )
                    utils.save_experience_data(
                        task_id=f"{task_id}_attempt_{i}_first",
                        experience_data=first_record,
                        data_dir=self.train_dir,
                    )

                # If not successful, do reflection and retry from the beginning
                if not success:
                    print(f"[ReflectGRPO] Failed attempt {i}, attempting reflection and retry from beginning...")
                    reflect_checklist, reflection_text = self.get_reflect(trajectory)
                    is_valid, is_perfect = utils.validate_reflect_report(
                        reflect_checklist, attempts
                    )

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
                                second_success,
                                second_predicted_answer,
                                second_ground_truth,
                                second_attempts,
                            ) = utils.second_rollout(self, guidance_prompt, trajectory, retry_step)

                            print(
                                f"[ReflectGRPO] Retry {i} - reward: {second_reward}, attempts: {second_attempts}, improved: {second_reward > reward}"
                            )

                            second_exp = self.model.convert_messages_to_experience(
                                distill_trajectory[:-1]
                            )
                            second_exp.reward = second_reward
                            second_exp.metrics = {
                                "success": 1.0 if second_success else 0.0,
                                "reward": second_reward,
                                "attempts": second_attempts,
                                "improved": 1.0 if second_reward > reward else 0.0,
                            }
                            exp_lst.append(second_exp)

                            if self.whether_save_data:
                                # Save second attempt data with reflection information
                                second_record = utils.create_experience_record(
                                    task_id=task_id,
                                    trajectory=second_trajectory,
                                    reward=second_reward,
                                    success=second_success,
                                    predicted_answer=second_predicted_answer,
                                    ground_truth=second_ground_truth,
                                    attempt_type="second",
                                    additional_metrics={
                                        "first_reward": reward,
                                        "improvement": second_reward > reward,
                                        "reward_difference": second_reward - reward,
                                        "reflection_text": reflection_text,
                                        "reflection_checklist": reflect_checklist,
                                        "guidance_prompt": guidance_prompt,
                                        "retry_from_beginning": True,
                                    },
                                )
                                utils.save_experience_data(
                                    task_id=f"{task_id}_attempt_{i}_second",
                                    experience_data=second_record,
                                    data_dir=self.train_dir,
                                )

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
