# -*- coding: utf-8 -*-
"""Critique-GRPO workflow for Alfworld environment.

This workflow implements the Critique-GRPO approach for agentic environments:
1. Generate n initial trajectories (environment interaction)
2. For each failed trajectory, generate a critique
3. For each critique, generate a refinement (re-execute with guidance)
4. Select the best refinement
5. Return n experiences: 1 refined (off-policy) + (n-1) initial (on-policy)
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from jinja2 import Environment, FileSystemLoader

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.envs.R3L.alfworld import utils
from trinity.common.workflows.workflow import WORKFLOWS, Task, Workflow


@WORKFLOWS.register_module("critique_grpo_alfworld_workflow")
class CritiqueGRPOAlfworldWorkflow(Workflow):
    """
    Critique-GRPO Workflow for Alfworld environment.

    Flow:
    1. Generate n initial trajectories via environment interaction
    2. For each failed initial, generate critique + refinement (re-execute)
    3. Select best refinement (prefer reward=1.0)
    4. Return n experiences:
       - Position 0: best refined (is_off_policy=True) if exists
       - Other positions: initial responses (is_off_policy=False)
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
        self.temperature = getattr(task.rollout_args, "temperature", 1.0)
        self.max_env_steps = 25
        self.max_tokens = 512
        self.max_critique_tokens = 1024
        self.task = task
        self.is_eval = task.is_eval
        self.whether_save_data = False

        # Create data directories
        self.data_dir = "critique_grpo_alfworld_data"
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

        self.alfworld_system_template = self.jinja_env.get_template("alfworld_system.j2")
        self.critique_template = self.jinja_env.get_template("critique.j2")

        print(f"Initializing CritiqueGRPOAlfworldWorkflow, temperature={self.temperature}")
        self.reset(task)

    def reset(self, task: Task):
        """Reset the workflow with a new task"""
        self.game_file_path = task.task_desc or task.raw_task.get("game_file", "")
        self.is_eval = task.is_eval
        self.temperature = getattr(task.rollout_args, "temperature", 1.0)
        self.task = task
        self.n = task.repeat_times

    def generate_critique(
        self, trajectory: List[Dict[str, str]], reward: float
    ) -> Tuple[Optional[str], Optional[Any]]:
        """Generate a critique for a failed trajectory."""
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
        """Generate a refinement by re-executing the task with guidance."""
        observation, info = env.reset()
        trajectory = []
        action_history = []

        # System prompt with critique guidance
        original_system_prompt = self.alfworld_system_template.render()
        guidance = f"""# Previous Attempt Analysis
{critique_text}

# Instructions
Use the above analysis to avoid similar mistakes. Focus on efficient task completion."""

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

            if responses[0].tokens.shape[0] >= 20480 - 512:
                return trajectory, default_reward, False, step + 1, False

            response_text = responses[0].response_text.strip()
            trajectory.append({"role": "assistant", "content": response_text})

            think, action, error_msg = utils.parse_response(response_text)
            if error_msg is not None:
                valid_format = False
                trajectory.append({"role": "user", "content": f"Feedback: {error_msg}"})
                return trajectory, default_reward, False, step + 1, valid_format

            observation, reward, done, info = env.step(action)
            if action not in admissible_actions:
                valid_format = False
                trajectory.append(
                    {"role": "user", "content": f"Feedback: Invalid action '{action}'"}
                )
                return trajectory, default_reward, False, step + 1, valid_format

            action_history.append(action)

            if len(action_history) >= 3 and all(
                a == action_history[-1] for a in action_history[-3:]
            ):
                trajectory.append(
                    {"role": "user", "content": "Feedback: Repeated action, task failed"}
                )
                return trajectory, default_reward, False, step + 1, False

            if done:
                break

        # Final feedback
        if reward >= 1.0:
            feedback = f"Task completed successfully (reward: {reward}/1.0)"
        else:
            feedback = f"Task not completed (reward: {reward}/1.0)"
        trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})

        return trajectory, reward, done, step + 1, valid_format

    def run(self) -> List[Experience]:
        """Run the Critique-GRPO workflow and return experiences."""

        if self.is_eval:
            return utils.eval_alfworld(self)

        env = utils.create_alfworld_environment(self.game_file_path)

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

                initial_results.append(
                    {
                        "trajectory": trajectory,
                        "reward": reward,
                        "exp": exp,
                    }
                )
            except Exception as e:
                print(f"[Critique-GRPO] Initial rollout {i+1} failed: {e}")
                initial_results.append(None)

        initial_results = [r for r in initial_results if r is not None]

        if not initial_results:
            return []

        # Step 2: For each failed initial, generate critique + refinement
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
                    env, critique_text
                )
                print(f"[Critique-GRPO] Refinement for initial {i+1} - reward: {ref_reward}")

                if ref_reward > result["reward"]:
                    # Create clean trajectory without guidance
                    clean_traj = []
                    for msg in ref_traj:
                        if msg["role"] == "system":
                            clean_traj.append(
                                {
                                    "role": "system",
                                    "content": self.alfworld_system_template.render(),
                                }
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

                    refinements.append(
                        {
                            "exp": ref_exp,
                            "reward": ref_reward,
                            "original_idx": i,
                        }
                    )
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

        # Pad if needed
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
