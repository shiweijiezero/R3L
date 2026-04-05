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
from trinity.common.workflows.envs.R3L.webshop import utils
from trinity.common.workflows.workflow import WORKFLOWS, Task, Workflow


@WORKFLOWS.register_module("R3L_webshop_w_o_credit_workflow")
class R3LWebshopWoCreditWorkflow(Workflow):
    """
    R3L workflow for webshop without credit assignment (ablation study)
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
        self.max_env_steps = 15
        self.max_tokens = 512
        self.max_reflect_tokens = 4096
        self.task = task
        self.is_eval = task.is_eval

        self.whether_save_data = False
        # Create data directories
        self.data_dir = f"R3L_webshop_data"
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
            # Try gymnasium first, fallback to gym
            import gym
            from web_agent_site.envs import WebAgentTextEnv  # noqa: F401

            # NOTE: Hosting the env require ~15GB CPU memory.
            # If you want easier env, you can set the num_products to 1000 or 100000.
            self.env = gym.make(
                "WebAgentTextEnv-v0",
                observation_mode="text_rich",
                num_products=None,
                human_goals=True,
            )
        except Exception as e:
            error_message = f"Error importing WebAgentTextEnv {str(e)}. Please make sure you have installed the web_agent_site package, following the instructions in https://github.com/princeton-nlp/WebShop"
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

        self.default_exp = Experience(
            tokens=torch.tensor([0, 0], dtype=torch.long),
            prompt_length=1,
            action_mask=torch.tensor([False], dtype=torch.bool),
            logprobs=torch.tensor([0.0], dtype=torch.float),
            metrics={
                "success": 0.0,
                "reward": -0.1,
            },
            reward=-0.1,  # Default minimum reward for webshop tasks
        )

        self.default_second_exp = Experience(
            tokens=torch.tensor([0, 0], dtype=torch.long),
            prompt_length=1,
            action_mask=torch.tensor([False], dtype=torch.bool),
            logprobs=torch.tensor([0.0], dtype=torch.float),
            metrics={
                "second_success": 0.0,
                "second_reward": -0.1,
            },
            reward=-0.1,
        )

        print(
            f"Initializing ExpLearnWebshopWorkflow with experience learning, temperature={self.temperature}"
        )
        self.reset(task)

    def reset(self, task: Task):
        """Reset the workflow with a new task"""
        self.session_id = int(task.task_desc or "0")
        self.is_eval = task.is_eval
        self.task = task
        self.n = task.repeat_times
        self.temperature = getattr(task.rollout_args, "temperature", 1.0)

    def get_reflect(
        self, trajectory: List[Dict[str, str]]
    ) -> tuple[Optional[Dict[str, Any]], Optional[str], Optional[Any]]:
        """
        Generates a comprehensive reflection report using a single, unified self-interrogation prompt.
        The model first assesses its own performance and then follows the appropriate reflection path.
        """
        # print("Generating reflection report using the unified self-interrogation prompt...")

        # 格式化轨迹以供LLM阅读
        formatted_trajectory = utils.format_trajectory_for_reflection(trajectory)

        # 使用Jinja2模板渲染反思提示
        reflect_prompt = self.reflection_template.render()

        # 调用模型并解析结果
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

            # print(f"raw reflection text: {reflection_text}")

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
            # print(f"Failed during unified reflection process: {e}")
            return None, None, None


    def run(self) -> List[Experience]:
        """Run the experience learning webshop workflow and return experiences"""

        if self.is_eval:
            # print("pass evaluation mode")
            # return [opmd_reflect_enhanced_restart_utils.generate_default_experience()]
            return self.eval_webshop()

        # Generate unique task ID using timestamp
        task_id = f"{str(self.task.batch_id).replace('/', '_')}_{self.task.task_id}"

        exp_lst = []
        for i in range(self.n // 2):  # 一半用于rollout，一半在此基础上进行反思再rollout
            try:
                trajectory, reward, done, steps, format_valid = utils.first_rollout(
                    self, self.env, self.session_id
                )
                print(f"[R3L w/o credit] First rollout - reward: {reward}, steps: {steps}")

                exp = self.model.convert_messages_to_experience(trajectory[:-1])
                exp.reward = reward
                # exp.info = {"valid": format_valid}
                # print(exp.info)
                exp.metrics = {
                    "success": 1.0 if reward >= 1.0 else 0.0,
                    "steps": steps,
                    "reward": reward,
                }
                # 设置eid
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

                # 对首次尝试进行反思
                reflect_checklist, reflection_text, reflect_exp = self.get_reflect(trajectory)
                is_valid, is_perfect = utils.validate_reflect_report(reflect_checklist, steps)

                if not is_valid or is_perfect:
                    # 如果第一次尝试的reward是1.0且反思给出完美，则记录反思exp
                    if reward >= 1.0 and is_perfect and reflect_exp is not None:
                        reflect_exp.reward = 1.0
                        # 设置eid
                        reflect_exp.eid.task = str(self.task.task_id) + f"_reflect_{i}"
                        reflect_exp.eid.run = len(exp_lst) + self.run_id_base
                        exp_lst.append(reflect_exp)

                else:
                    guidance_prompt = utils.reflect_report_to_guidance_prompt(reflect_checklist)
                    # Extract retry_step from validated reflection report (top-level field in alfworld schema)
                    retry_step = reflect_checklist.get("retry_from_step", 0)
                    print(f"[R3L w/o credit] Pivot point (retry_from_step): {retry_step}")

                    try:
                        (
                            distill_trajectory,
                            second_trajectory,
                            second_reward,
                            second_done,
                            second_steps,
                            second_format_valid,
                        ) = utils.second_rollout(
                            self, self.env, self.session_id, guidance_prompt, trajectory, retry_step
                        )
                        print(
                            f"[R3L w/o credit] Second rollout - reward: {second_reward}, steps: {second_steps}, improve: {second_reward > reward}"
                        )
                        second_exp = self.model.convert_messages_to_experience(
                            distill_trajectory[:-1]
                        )

                        # Adjust action_mask to exclude retry prefix from training
                        if retry_step > 0:
                            self._adjust_action_mask_for_retry(second_exp, retry_step)
                            # Also adjust first rollout exp for fair comparison
                            # Find and modify the exp that was already added to exp_lst
                            for existing_exp in exp_lst:
                                if existing_exp.eid.run == exp_run_id:
                                    self._adjust_action_mask_for_retry(existing_exp, retry_step)
                                    break

                        second_exp.reward = second_reward
                        # second_exp.info = {"valid": second_format_valid}
                        second_exp.metrics = {
                            "second_success": 1.0 if second_reward >= 1.0 else 0.0,
                            "second_steps": second_steps,
                            "second_reward": second_reward,
                            "second_improve": 1.0 if second_reward > reward else 0.0,
                            "second_reward_diff": second_reward - reward,
                            "pivot_point": retry_step,
                        }
                        # 设置eid
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

                        # 如果第二次尝试的分数高于第一次，或第二次是满分情况下步数更少，则记录反思和重试数据
                        if (second_reward > reward and second_reward >= 1.0) or (
                            second_reward >= 1.0 and second_steps < steps
                        ):
                            # 将反思数据转换为exp
                            # reflect_exp.reward = second_reward - reward
                            reflect_exp.reward = 1.0
                            # 设置eid
                            reflect_exp.eid.task = str(self.task.task_id) + f"_reflect_{i}"
                            reflect_exp.eid.run = len(exp_lst) + self.run_id_base
                            exp_lst.append(reflect_exp)

                            # 将重试数据转换为exp
                            retry_exp = self.model.convert_messages_to_experience(
                                second_trajectory[:-1]
                            )

                            # Adjust action_mask to exclude retry prefix from training
                            # NOTE: Credit assignment removed for ablation study
                            # No adjustment to action_mask based on retry_step

                            # retry_exp.reward = second_reward - reward
                            retry_exp.reward = 1.0
                            # 设置eid
                            retry_exp.eid.task = str(self.task.task_id) + f"_retry_{i}"
                            retry_exp.eid.run = len(exp_lst) + self.run_id_base
                            exp_lst.append(retry_exp)

                            # print
                            print("Reflection and retry led to improvement, recording both...")
                    except Exception:
                        pass
            except Exception:
                pass

        return exp_lst

    def set_repeat_times(self, repeat_times, run_id_base):
        self.repeat_times = repeat_times
        self.run_id_base = run_id_base
        self.n = repeat_times
