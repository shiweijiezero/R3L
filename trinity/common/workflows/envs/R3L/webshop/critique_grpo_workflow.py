# -*- coding: utf-8 -*-
"""Critique-GRPO workflow for WebShop environment."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from jinja2 import Environment, FileSystemLoader

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.envs.R3L.webshop import utils
from trinity.common.workflows.workflow import WORKFLOWS, Task, Workflow


@WORKFLOWS.register_module("critique_grpo_webshop_workflow")
class CritiqueGRPOWebShopWorkflow(Workflow):
    """Critique-GRPO Workflow for WebShop environment."""

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
        self.temperature = getattr(task.rollout_args, "temperature", 1.0)
        self.max_env_steps = 15
        self.max_tokens = 512
        self.max_critique_tokens = 1024
        self.task = task
        self.is_eval = task.is_eval
        self.whether_save_data = False

        self.data_dir = "critique_grpo_webshop_data"
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
                sys.path.append("/path/to/webshop")
            import gym
            from web_agent_site.envs import WebAgentTextEnv  # noqa: F401

            # NOTE: Hosting the env requires ~15GB CPU memory.
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

        prompts_dir = Path(__file__).parent / "prompts"
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(prompts_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.webshop_system_template = self.jinja_env.get_template("webshop_system.j2")
        self.critique_template = self.jinja_env.get_template("critique.j2")

        print(f"Initializing CritiqueGRPOWebShopWorkflow, temperature={self.temperature}")
        self.reset(task)

    def reset(self, task: Task):
        self.task_desc = task.task_desc or "0"
        self.session_id = int(task.task_desc or "0")
        self.is_eval = task.is_eval
        self.task = task
        self.n = task.repeat_times
        self.temperature = getattr(task.rollout_args, "temperature", 1.0)

    def generate_critique(
        self, trajectory: List[Dict[str, str]], reward: float
    ) -> Tuple[Optional[str], Optional[Any]]:
        formatted_trajectory = utils.format_trajectory_for_reflection(trajectory)

        critique_prompt = self.critique_template.render(
            trajectory=formatted_trajectory,
            reward=reward,
        )

        try:
            responses = self.model.chat(
                [{"role": "user", "content": critique_prompt}],
                n=1,
                temperature=0.7,
                max_tokens=self.max_critique_tokens,
            )
            return responses[0].response_text.strip(), responses[0]
        except Exception as e:
            print(f"[Critique-GRPO] Critique generation failed: {e}")
            return None, None

    def generate_refinement(
        self, critique_text: str
    ) -> Tuple[List[Dict[str, str]], float, bool, int, bool]:
        """Generate a refinement rollout with critique guidance."""
        self.env.reset(session=self.session_id)
        observation = self.env.observation
        trajectory = []
        action_history = []

        original_system_prompt = self.webshop_system_template.render()
        guidance = f"""# Previous Attempt Analysis
{critique_text}

# Instructions
Use the above analysis to avoid similar mistakes."""

        merged_system_prompt = f"{original_system_prompt}\n\n{guidance}"
        trajectory.append({"role": "system", "content": merged_system_prompt})

        default_reward = -0.1
        done = False
        reward = default_reward
        valid_format = True
        step = 0

        for step in range(self.max_env_steps):
            available_actions = self.env.get_available_actions()

            trajectory.append(
                {
                    "role": "user",
                    "content": utils.format_observation(observation, available_actions),
                }
            )

            responses = self.model.chat(
                trajectory,
                n=1,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            if responses[0].tokens.shape[0] >= 20480 - self.max_tokens:
                return trajectory, default_reward, False, step + 1, False

            response_text = responses[0].response_text.strip()
            trajectory.append({"role": "assistant", "content": response_text})

            think, action = utils.parse_response(response_text)
            if action is None:
                valid_format = False
                feedback = "Invalid response format, missing valid <think> or <action> tags"
                trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
                return trajectory, default_reward, False, step + 1, valid_format

            # Check for consecutive action repetition
            action_history.append(action)
            if len(action_history) > 2:
                action_history.pop(0)

            if len(action_history) >= 2 and all(
                a == action_history[0] for a in action_history
            ) and "next" not in action.lower() and "prev" not in action.lower() and "search" not in action.lower():
                feedback = f"Repeated invalid action {action} multiple times, shopping task failed"
                trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
                valid_format = False
                return trajectory, default_reward, False, step + 1, valid_format

            # Validate and execute action
            action_valid, error_msg = utils.validate_action(action, available_actions)
            if action_valid:
                observation, reward, done, info = self.env.step(action)
            else:
                observation, reward, done = error_msg, default_reward, False

            if done:
                break

        # Generate feedback
        if reward >= 1.0:
            feedback = f"Shopping task completed successfully (reward: {reward}/1.0)"
        else:
            feedback = f"Shopping task not completed (reward: {reward}/1.0)"

        trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
        return trajectory, reward, done, step + 1, valid_format

    def run(self) -> List[Experience]:
        if self.is_eval:
            return utils.eval_webshop(self)

        # Step 1: Generate n initial trajectories
        initial_results = []
        for i in range(self.n):
            try:
                trajectory, reward, done, steps, valid = utils.first_rollout(
                    self, self.env, self.session_id
                )
                print(f"[Critique-GRPO] Initial {i+1}/{self.n} - reward: {reward}")

                exp = self.model.convert_messages_to_experience(trajectory[:-1])
                exp.reward = reward
                exp.metrics = {
                    "success": 1.0 if reward >= 1.0 else 0.0,
                    "steps": steps,
                    "reward": reward,
                }
                exp.eid.task = str(self.task.task_id) + "_initial"
                exp.eid.run = i + self.run_id_base

                initial_results.append({"trajectory": trajectory, "reward": reward, "exp": exp})
            except Exception as e:
                print(f"[Critique-GRPO] Initial rollout {i+1} failed: {e}")

        initial_results = [r for r in initial_results if r is not None]
        if not initial_results:
            return []

        # Step 2: Generate refinements for failed ones
        refinements = []
        for i, result in enumerate(initial_results):
            if result["reward"] >= 1.0:
                continue
            try:
                critique_text, _ = self.generate_critique(result["trajectory"], result["reward"])
                if critique_text is None:
                    continue

                print(f"[Critique-GRPO] Generated critique for initial {i+1}")

                # Re-execute with guidance
                ref_traj, ref_reward, ref_done, ref_steps, ref_valid = self.generate_refinement(
                    critique_text
                )
                print(f"[Critique-GRPO] Refinement for initial {i+1} - reward: {ref_reward}")

                if ref_reward > result["reward"]:
                    # Create clean trajectory without guidance
                    clean_traj = []
                    for msg in ref_traj:
                        if msg["role"] == "system":
                            clean_traj.append(
                                {"role": "system", "content": self.webshop_system_template.render()}
                            )
                        else:
                            clean_traj.append(msg)

                    ref_exp = self.model.convert_messages_to_experience(clean_traj[:-1])
                    ref_exp.reward = ref_reward
                    ref_exp.metrics = {
                        "refined_success": 1.0 if ref_reward >= 1.0 else 0.0,
                        "refined_reward": ref_reward,
                        "refined_steps": ref_steps,
                        "improvement": ref_reward - result["reward"],
                    }
                    ref_exp.eid.task = str(self.task.task_id) + "_refined"
                    ref_exp.eid.run = i + self.run_id_base + 100

                    refinements.append({"exp": ref_exp, "reward": ref_reward, "original_idx": i})
            except Exception as e:
                print(f"[Critique-GRPO] Refinement failed: {e}")

        # Step 3: Select best refinement
        best_refinement = None
        if refinements:
            refinements.sort(key=lambda x: x["reward"], reverse=True)
            for ref in refinements:
                if ref["reward"] >= 1.0:
                    best_refinement = ref
                    break
            if best_refinement is None:
                best_refinement = refinements[0]

        # Step 4: Construct final experience list
        exp_lst = []
        if best_refinement is not None:
            ref_exp = best_refinement["exp"]
            ref_exp.info["is_off_policy"] = True
            exp_lst.append(ref_exp)
            print(
                f"[Critique-GRPO] Using refined response (reward={best_refinement['reward']}) as off-policy sample"
            )

            for i, result in enumerate(initial_results):
                if i == best_refinement["original_idx"]:
                    continue
                if len(exp_lst) >= self.n:
                    break
                result["exp"].info["is_off_policy"] = False
                exp_lst.append(result["exp"])
        else:
            print("[Critique-GRPO] No improvement from refinement, using all initial responses")
            for result in initial_results:
                if len(exp_lst) >= self.n:
                    break
                result["exp"].info["is_off_policy"] = False
                exp_lst.append(result["exp"])

        while len(exp_lst) < self.n and initial_results:
            for result in initial_results:
                if len(exp_lst) >= self.n:
                    break
                result["exp"].info["is_off_policy"] = False
                exp_lst.append(result["exp"])

        off_policy_count = sum(1 for exp in exp_lst if exp.info.get("is_off_policy", False))
        print(f"[Critique-GRPO Summary] {len(exp_lst)} experiences: {off_policy_count} off-policy")
        return exp_lst

    def set_repeat_times(self, repeat_times, run_id_base):
        self.repeat_times = repeat_times
        self.run_id_base = run_id_base
        self.n = repeat_times
