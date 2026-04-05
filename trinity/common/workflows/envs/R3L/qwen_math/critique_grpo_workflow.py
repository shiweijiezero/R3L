# -*- coding: utf-8 -*-
"""Critique-GRPO workflow for Qwen Math problem solving.

This workflow implements the Critique-GRPO approach:
1. Generate n initial responses
2. For each failed response, generate a critique and refinement
3. Select the best refinement
4. Return n experiences: 1 refined (off-policy) + (n-1) initial (on-policy)
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from jinja2 import Environment, FileSystemLoader

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.envs.R3L.qwen_math import utils
from trinity.common.workflows.workflow import WORKFLOWS, Task, Workflow


@WORKFLOWS.register_module("critique_grpo_qwen_math_workflow")
class CritiqueGRPOQwenMathWorkflow(Workflow):
    """
    Critique-GRPO Workflow for Qwen Math problem solving.

    Flow:
    1. Generate n initial responses (via first_rollout)
    2. For each failed initial, generate critique + refinement
    3. Select best refinement (prefer reward=1.0)
    4. Return n experiences:
       - Position 0: best refined (is_off_policy=True) if exists, else initial
       - Other positions: remaining initial responses (is_off_policy=False)
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
        self.max_critique_tokens = 2048
        self.task = task
        self.is_eval = task.is_eval
        self.whether_save_data = True

        # Create data directories
        self.data_dir = "critique_grpo_qwen_math_data"
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

        # Cache templates
        self.qwen_system_template = self.jinja_env.get_template("math_system.j2")
        self.critique_template = self.jinja_env.get_template("critique.j2")

        print(f"Initializing CritiqueGRPOQwenMathWorkflow, temperature={self.temperature}")
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

    def generate_critique(
        self, trajectory: List[Dict[str, str]], reward: float
    ) -> Tuple[Optional[str], Optional[Any]]:
        """
        Generate a critique for a failed trajectory.

        Args:
            trajectory: The conversation trajectory
            reward: The reward obtained (0.0 for failure)

        Returns:
            Tuple of (critique_text, critique_response)
        """
        # Format trajectory for critique
        formatted_trajectory = utils.format_trajectory_for_reflection(trajectory)

        # Render critique prompt
        try:
            critique_prompt = self.critique_template.render(
                trajectory=formatted_trajectory,
                reward=reward,
            )
        except Exception as e:
            print(f"[Critique-GRPO-Qwen] Failed to render critique template: {e}")
            # Fallback to simple critique prompt
            critique_prompt = f"""Analyze the following math problem solving attempt:

{formatted_trajectory}

The attempt resulted in reward: {reward}

Please provide a detailed critique identifying:
1. What went wrong in the reasoning
2. Key conceptual errors
3. Specific suggestions for improvement

Output your critique in a clear, structured format."""

        try:
            responses = self.model.chat(
                [
                    {
                        "role": "system",
                        "content": "You are an expert math tutor providing constructive feedback.",
                    },
                    {"role": "user", "content": critique_prompt},
                ],
                n=1,
                temperature=0.7,  # Lower temperature for more focused critique
                max_tokens=self.max_critique_tokens,
            )
            critique_text = responses[0].response_text.strip()
            return critique_text, responses[0]
        except Exception as e:
            print(f"[Critique-GRPO-Qwen] Critique generation failed: {e}")
            return None, None

    def generate_refinement(
        self, critique_text: str
    ) -> Tuple[List[Dict[str, str]], float, bool, str, str, int]:
        """
        Generate a refinement based on the critique.

        This is a complete re-execution of the math problem with guidance.

        Args:
            critique_text: The critique from previous attempt

        Returns:
            Same as first_rollout: (trajectory, reward, success, predicted_answer, ground_truth, attempts)
        """
        trajectory = []

        # System prompt with critique guidance
        original_system_prompt = self.qwen_system_template.render()
        guidance = f"""# Previous Attempt Analysis
{critique_text}

# Instructions
Use the above analysis to avoid similar mistakes. Focus on the key improvements suggested."""

        merged_system_prompt = f"{original_system_prompt}\n\n{guidance}"
        trajectory.append({"role": "system", "content": merged_system_prompt})

        # Add user prompt
        problem_prompt = (
            self.prompt if self.prompt else "Please solve the given mathematical problem."
        )
        formatted_prompt = utils.format_qwen_math_prompt(problem_prompt, attempt=0)
        trajectory.append({"role": "user", "content": formatted_prompt})

        final_reward = 0.0
        final_success = False
        final_predicted_answer = ""
        attempt_count = 0

        # Try up to 3 attempts
        for attempt in range(self.max_attempts):
            attempt_count = attempt + 1

            responses = self.model.chat(
                trajectory,
                n=1,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # Check token limit
            if responses[0].tokens.shape[0] >= 20480 - 4096:
                return (
                    trajectory,
                    final_reward,
                    final_success,
                    final_predicted_answer,
                    self.ground_truth,
                    attempt_count,
                )

            response_text = responses[0].response_text.strip()
            trajectory.append({"role": "assistant", "content": response_text})

            # Parse response using boxed format
            predicted_answer = utils.parse_boxed_response(response_text)

            if predicted_answer is None:
                feedback = "I could not find an answer in \\boxed{} format. Please put your final answer within \\boxed{}."
                formatted_feedback = utils.format_qwen_math_prompt(
                    "", attempt=attempt_count, feedback=feedback
                )
                trajectory.append({"role": "user", "content": formatted_feedback})
                continue

            # Verify answer
            is_correct = utils.my_math_verify(predicted_answer, self.ground_truth)

            if is_correct:
                final_reward = 1.0
                final_success = True
                final_predicted_answer = predicted_answer
                feedback = f"Correct! Your answer {predicted_answer} matches the expected answer."
                trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
                break
            else:
                if attempt < self.max_attempts - 1:
                    feedback = f"Incorrect. Your answer {predicted_answer} does not match. Please try again."
                    formatted_feedback = utils.format_qwen_math_prompt(
                        "", attempt=attempt_count, feedback=feedback
                    )
                    trajectory.append({"role": "user", "content": formatted_feedback})
                else:
                    feedback = f"Incorrect. Your answer {predicted_answer} does not match the expected answer. Maximum attempts reached."
                    trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
                final_predicted_answer = predicted_answer

        return (
            trajectory,
            final_reward,
            final_success,
            final_predicted_answer,
            self.ground_truth,
            attempt_count,
        )

    def run(self) -> List[Experience]:
        """Run the Critique-GRPO workflow and return experiences."""

        if self.is_eval:
            return utils.eval_qwen_math(self)

        task_id = f"{str(self.task.batch_id).replace('/', '_')}_{self.task.task_id}"

        # Step 1: Generate n initial responses
        initial_results = []  # List of (trajectory, reward, success, exp)
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
                print(f"[Critique-GRPO-Qwen] Initial {i+1}/{self.n} - reward: {reward}")

                exp = self.model.convert_messages_to_experience(trajectory[:-1])
                exp.reward = reward
                exp.metrics = {
                    "success": 1.0 if success else 0.0,
                    "reward": reward,
                    "attempts": attempts,
                }
                exp.eid.task = str(self.task.task_id) + "_initial"
                exp.eid.run = i + self.run_id_base

                initial_results.append(
                    {
                        "trajectory": trajectory,
                        "reward": reward,
                        "success": success,
                        "exp": exp,
                        "predicted_answer": predicted_answer,
                    }
                )
            except Exception as e:
                print(f"[Critique-GRPO-Qwen] Initial rollout {i+1} failed: {e}")
                initial_results.append(None)

        # Remove failed rollouts
        initial_results = [r for r in initial_results if r is not None]

        if not initial_results:
            print("[Critique-GRPO-Qwen] All initial rollouts failed")
            return []

        # Step 2: For each failed initial, generate critique + refinement
        refinements = []  # List of (exp, reward) for successful refinements
        for i, result in enumerate(initial_results):
            if result["reward"] >= 1.0:
                # Already successful, no need for refinement
                continue

            try:
                # Generate critique
                critique_text, _ = self.generate_critique(result["trajectory"], result["reward"])
                if critique_text is None:
                    continue

                print(f"[Critique-GRPO-Qwen] Generated critique for initial {i+1}")

                # Generate refinement (complete re-execution with guidance)
                (
                    ref_traj,
                    ref_reward,
                    ref_success,
                    ref_pred,
                    _,
                    ref_attempts,
                ) = self.generate_refinement(critique_text)
                print(
                    f"[Critique-GRPO-Qwen] Refinement for initial {i+1} - reward: {ref_reward}"
                )

                if ref_reward > result["reward"]:
                    # Create refinement experience (off-policy)
                    # For refinement, we use the trajectory without guidance system prompt
                    # to create a clean experience for training
                    clean_traj = []
                    for msg in ref_traj:
                        if msg["role"] == "system":
                            # Use original system prompt without guidance
                            clean_traj.append(
                                {
                                    "role": "system",
                                    "content": self.qwen_system_template.render(),
                                }
                            )
                        else:
                            clean_traj.append(msg)

                    ref_exp = self.model.convert_messages_to_experience(clean_traj[:-1])
                    ref_exp.reward = ref_reward
                    ref_exp.metrics = {
                        "refined_success": 1.0 if ref_success else 0.0,
                        "refined_reward": ref_reward,
                        "refined_attempts": ref_attempts,
                        "improvement": ref_reward - result["reward"],
                    }
                    ref_exp.eid.task = str(self.task.task_id) + "_refined"
                    ref_exp.eid.run = i + self.run_id_base + 100  # Offset to avoid collision

                    refinements.append(
                        {
                            "exp": ref_exp,
                            "reward": ref_reward,
                            "original_idx": i,
                        }
                    )
            except Exception as e:
                print(
                    f"[Critique-GRPO-Qwen] Critique/refinement for initial {i+1} failed: {e}"
                )

        # Step 3: Select best refinement (prefer reward=1.0)
        best_refinement = None
        if refinements:
            # Sort by reward descending
            refinements.sort(key=lambda x: x["reward"], reverse=True)
            # Prefer reward=1.0, otherwise take highest
            for ref in refinements:
                if ref["reward"] >= 1.0:
                    best_refinement = ref
                    break
            if best_refinement is None:
                best_refinement = refinements[0]

        # Step 4: Construct final experience list
        exp_lst = []

        if best_refinement is not None:
            # First position: refined experience (off-policy)
            ref_exp = best_refinement["exp"]
            ref_exp.info["is_off_policy"] = True
            exp_lst.append(ref_exp)
            print(
                f"[Critique-GRPO-Qwen] Using refined response (reward={best_refinement['reward']}) as off-policy sample"
            )

            # Remaining positions: initial responses (on-policy), excluding the one that was refined
            for i, result in enumerate(initial_results):
                if i == best_refinement["original_idx"]:
                    continue  # Skip the refined one
                if len(exp_lst) >= self.n:
                    break
                result["exp"].info["is_off_policy"] = False
                exp_lst.append(result["exp"])
        else:
            # No refinement available, all are on-policy
            print(
                "[Critique-GRPO-Qwen] No improvement from refinement, using all initial responses"
            )
            for result in initial_results:
                if len(exp_lst) >= self.n:
                    break
                result["exp"].info["is_off_policy"] = False
                exp_lst.append(result["exp"])

        # Pad to n experiences if needed (should rarely happen)
        while len(exp_lst) < self.n and initial_results:
            # Duplicate some experiences
            for result in initial_results:
                if len(exp_lst) >= self.n:
                    break
                result["exp"].info["is_off_policy"] = False
                exp_lst.append(result["exp"])

        # Log summary
        off_policy_count = sum(1 for exp in exp_lst if exp.info.get("is_off_policy", False))
        on_policy_count = len(exp_lst) - off_policy_count
        print(
            f"[Critique-GRPO-Qwen Summary] {len(exp_lst)} experiences: {off_policy_count} off-policy, {on_policy_count} on-policy"
        )

        return exp_lst

    def set_repeat_times(self, repeat_times, run_id_base):
        self.repeat_times = repeat_times
        self.run_id_base = run_id_base
        self.n = repeat_times
