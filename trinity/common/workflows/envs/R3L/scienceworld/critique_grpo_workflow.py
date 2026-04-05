# -*- coding: utf-8 -*-
"""Critique-GRPO workflow for ScienceWorld environment."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from jinja2 import Environment, FileSystemLoader

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.envs.R3L.scienceworld import utils
from trinity.common.workflows.workflow import WORKFLOWS, Task, Workflow


@WORKFLOWS.register_module("critique_grpo_scienceworld_workflow")
class CritiqueGRPOScienceWorldWorkflow(Workflow):
    """Critique-GRPO Workflow for ScienceWorld environment."""

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
        self.max_env_steps = 30
        self.max_tokens = 16384
        self.max_critique_tokens = 1024
        self.task = task
        self.is_eval = task.is_eval
        self.whether_save_data = False

        self.data_dir = "critique_grpo_scienceworld_data"
        self.eval_dir = os.path.join(self.data_dir, "eval")
        self.train_dir = os.path.join(self.data_dir, "train")
        os.makedirs(self.eval_dir, exist_ok=True)
        os.makedirs(self.train_dir, exist_ok=True)

        prompts_dir = Path(__file__).parent / "prompts"
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(prompts_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.sciworld_system_template = self.jinja_env.get_template("sciworld_system.j2")
        self.critique_template = self.jinja_env.get_template("critique.j2")

        print(f"Initializing CritiqueGRPOScienceWorldWorkflow, temperature={self.temperature}")
        self.reset(task)

    def reset(self, task: Task):
        self.task_desc = task.task_desc or "0"
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
        self, env, critique_text: str
    ) -> Tuple[List[Dict[str, str]], float, bool, int, bool]:
        observation, info = env.reset()
        trajectory = []
        action_history = []

        original_system_prompt = self.sciworld_system_template.render()
        guidance = f"""# Previous Attempt Analysis
{critique_text}

# Instructions
Use the above analysis to avoid similar mistakes."""

        merged_system_prompt = f"{original_system_prompt}\n\n{guidance}"
        trajectory.append({"role": "system", "content": merged_system_prompt})

        default_reward = 0.0
        done = False
        reward = default_reward
        valid_format = True

        task_description = utils.extract_task_description(observation)

        for step in range(self.max_env_steps):
            admissible_actions = (
                info.get("admissible_commands", []) if isinstance(info, dict) else []
            )

            trajectory.append(
                {
                    "role": "user",
                    "content": utils.format_observation(
                        current_observation=observation,
                        task_description=task_description,
                        current_step=step,
                        action_history=action_history,
                        admissible_actions=admissible_actions,
                    ),
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

            think, action, error_msg = utils.parse_response(response_text)
            if error_msg is not None:
                trajectory.append({"role": "user", "content": f"Feedback: {error_msg}"})
                return trajectory, default_reward, False, step + 1, False

            observation, reward, done, info = env.step(action)
            reward = reward / 100.0  # ScienceWorld returns 0-100

            if action not in admissible_actions:
                trajectory.append({"role": "user", "content": f"Feedback: Invalid action"})
                return trajectory, default_reward, False, step + 1, False

            action_history.append(action)

            if len(action_history) >= 3 and all(
                a == action_history[-1] for a in action_history[-3:]
            ):
                trajectory.append({"role": "user", "content": "Feedback: Repeated action"})
                return trajectory, default_reward, False, step + 1, False

            if done:
                break

        feedback = f"Task completed with reward: {reward}"
        trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
        return trajectory, reward, done, step + 1, valid_format

    def run(self) -> List[Experience]:
        if self.is_eval:
            return utils.eval_sciworld(self)

        env = utils.create_sciworld_environment(self.task_desc)

        # Step 1: Generate n initial trajectories
        initial_results = []
        for i in range(self.n):
            try:
                trajectory, reward, done, steps, valid = utils.first_rollout(self, env)
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

                ref_traj, ref_reward, ref_done, ref_steps, ref_valid = self.generate_refinement(
                    env, critique_text
                )
                print(f"[Critique-GRPO] Refinement for initial {i+1} - reward: {ref_reward}")

                if ref_reward > result["reward"]:
                    clean_traj = []
                    for msg in ref_traj:
                        if msg["role"] == "system":
                            clean_traj.append(
                                {
                                    "role": "system",
                                    "content": self.sciworld_system_template.render(),
                                }
                            )
                        else:
                            clean_traj.append(msg)

                    ref_exp = self.model.convert_messages_to_experience(clean_traj[:-1])
                    ref_exp.reward = ref_reward
                    ref_exp.metrics = {
                        "refined_reward": ref_reward,
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
            best_refinement["exp"].info["is_off_policy"] = True
            exp_lst.append(best_refinement["exp"])
            for i, result in enumerate(initial_results):
                if i == best_refinement["original_idx"]:
                    continue
                if len(exp_lst) >= self.n:
                    break
                result["exp"].info["is_off_policy"] = False
                exp_lst.append(result["exp"])
        else:
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
