# -*- coding: utf-8 -*-
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from jinja2 import Environment, FileSystemLoader
from math_verify import parse, verify

from trinity.common.experience import Experience
from trinity.utils.eval_utils import find_boxed_answer


def first_rollout(self) -> tuple[List[Dict[str, str]], float, bool, str, str, int]:
    """Run math problem solving with multiple attempts (max 3 attempts)"""
    trajectory = []

    # Add system prompt - use Qwen official format
    system_prompt = self.qwen_system_template.render()
    trajectory.append({"role": "system", "content": system_prompt})

    # Add user prompt - just the problem, no verbose format instructions
    problem_prompt = self.prompt if self.prompt else "Please solve the given mathematical problem."
    formatted_prompt = format_qwen_math_prompt(problem_prompt, attempt=0)
    trajectory.append({"role": "user", "content": formatted_prompt})

    final_reward = 0.0
    final_success = False
    final_predicted_answer = ""
    attempt_count = 0

    # Try up to max_attempts
    for attempt in range(self.max_attempts):
        attempt_count = attempt + 1

        # Get model response
        responses = self.model.chat(
            trajectory,
            n=1,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Check if tokens exceed limit
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

        # Parse answer from \boxed{}
        predicted_answer = parse_boxed_response(response_text)

        if predicted_answer is None:
            # Could not find \boxed{} answer
            feedback = (
                "I could not find an answer in \\boxed{} format."
                " Please put your final answer within \\boxed{}."
            )
            formatted_feedback = format_qwen_math_prompt(
                "", attempt=attempt_count, feedback=feedback
            )
            trajectory.append({"role": "user", "content": formatted_feedback})
            continue

        # Verify answer
        is_correct = my_math_verify(predicted_answer, self.ground_truth)

        if is_correct:
            final_reward = 1.0
            final_success = True
            final_predicted_answer = predicted_answer
            print(
                f"[First Rollout] Attempt {attempt_count} - Correct answer! Reward: {final_reward}"
            )
            feedback = f"Correct! Your answer {predicted_answer} matches the expected answer."
            trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
            break
        else:
            # Wrong answer
            print(
                f"[First Rollout] Attempt {attempt_count} - Incorrect answer: {predicted_answer} (Expected: {self.ground_truth})"
            )
            if attempt < self.max_attempts - 1:
                feedback = (
                    f"Incorrect. Your answer {predicted_answer} does not match. Please try again."
                )
                formatted_feedback = format_qwen_math_prompt(
                    "", attempt=attempt_count, feedback=feedback
                )
                trajectory.append({"role": "user", "content": formatted_feedback})
            else:
                # Last attempt
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


def second_rollout(
    self,
    guidance_prompt: str,
    first_trajectory: List[Dict[str, str]],
    retry_step: int = 0,
) -> tuple[List[Dict[str, str]], List[Dict[str, str]], float, bool, str, str, int]:
    """
    Performs rollout with guidance from reflection.
    For math problems, we typically start from the beginning with guidance.
    """
    trajectory = []
    distill_trajectory = []

    # Prepare system prompts
    original_system_prompt = self.qwen_system_template.render()

    # Starting from beginning with guidance
    merged_system_prompt = (
        f"{original_system_prompt}\n\n# Previous Attempt Analysis & Guidance\n{guidance_prompt}"
    )
    trajectory.append({"role": "system", "content": merged_system_prompt})
    distill_trajectory.append({"role": "system", "content": original_system_prompt})

    # Add user prompt - just the problem, no verbose format instructions
    problem_prompt = self.prompt if self.prompt else "Please solve the given mathematical problem."
    formatted_prompt = format_qwen_math_prompt(problem_prompt, attempt=0)
    trajectory.append({"role": "user", "content": formatted_prompt})
    distill_trajectory.append({"role": "user", "content": formatted_prompt})

    final_reward = 0.0
    final_success = False
    final_predicted_answer = ""
    attempt_count = 0

    # Try up to max_attempts
    for attempt in range(self.max_attempts):
        attempt_count = attempt + 1

        # Get model response with guidance
        responses = self.model.chat(
            trajectory,
            n=1,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Check if tokens exceed limit
        if responses[0].tokens.shape[0] >= 20480 - 4096:
            return (
                distill_trajectory,
                trajectory,
                final_reward,
                final_success,
                final_predicted_answer,
                self.ground_truth,
                attempt_count,
            )

        response_text = responses[0].response_text.strip()
        trajectory.append({"role": "assistant", "content": response_text})
        distill_trajectory.append({"role": "assistant", "content": response_text})

        # Parse answer from \boxed{}
        predicted_answer = parse_boxed_response(response_text)

        if predicted_answer is None:
            # Could not find \boxed{} answer
            feedback = (
                "I could not find an answer in \\boxed{} format."
                " Please put your final answer within \\boxed{}."
            )
            formatted_feedback = format_qwen_math_prompt(
                "", attempt=attempt_count, feedback=feedback
            )
            trajectory.append({"role": "user", "content": formatted_feedback})
            distill_trajectory.append({"role": "user", "content": formatted_feedback})
            continue

        # Verify answer
        is_correct = my_math_verify(predicted_answer, self.ground_truth)

        if is_correct:
            final_reward = 1.0
            final_success = True
            final_predicted_answer = predicted_answer
            print(
                f"[Second Rollout] Attempt {attempt_count} - Correct answer! Reward: {final_reward}"
            )
            feedback = f"Correct! Your answer {predicted_answer} matches the expected answer."
            trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
            distill_trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
            break
        else:
            # Wrong answer
            print(
                f"[Second Rollout] Attempt {attempt_count} - Incorrect answer: {predicted_answer} (Expected: {self.ground_truth})"
            )
            if attempt < self.max_attempts - 1:
                feedback = (
                    f"Incorrect. Your answer {predicted_answer} does not match. Please try again."
                )
                formatted_feedback = format_qwen_math_prompt(
                    "", attempt=attempt_count, feedback=feedback
                )
                trajectory.append({"role": "user", "content": formatted_feedback})
                distill_trajectory.append({"role": "user", "content": formatted_feedback})
            else:
                # Last attempt
                feedback = f"Incorrect. Your answer {predicted_answer} does not match the expected answer. Maximum attempts reached."
                trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
                distill_trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
            final_predicted_answer = predicted_answer

    return (
        distill_trajectory,
        trajectory,
        final_reward,
        final_success,
        final_predicted_answer,
        self.ground_truth,
        attempt_count,
    )


def eval_qwen_math(self) -> List[Experience]:
    """Evaluate a single math problem"""
    print("[Eval] Starting evaluation...")
    try:
        trajectory, reward, success, predicted_answer, ground_truth, attempts = first_rollout(self)
        exp = self.model.convert_messages_to_experience(trajectory[:-1])
        exp.reward = reward
        exp.metrics = {
            "success": 1.0 if success else 0.0,
            "reward": reward,
            "attempts": attempts,
        }
        print(f"[Eval] Completed - Reward: {reward}, Success: {success}, Attempts: {attempts}")
        print(f"[Eval] Predicted: {predicted_answer}, Ground Truth: {ground_truth}")

        if self.whether_save_data:
            # Save evaluation data
            eval_task_id = f"{str(self.task.batch_id).replace('/', '_')}_{self.task.task_id}"
            eval_record = create_experience_record(
                task_id=eval_task_id,
                trajectory=trajectory,
                reward=reward,
                success=success,
                predicted_answer=predicted_answer,
                ground_truth=ground_truth,
                attempt_type="evaluation",
            )
            save_experience_data(
                task_id=f"{eval_task_id}_eval", experience_data=eval_record, data_dir=self.eval_dir
            )
    except Exception as e:
        print(f"[Eval] Evaluation failed - Error: {str(e)}")
        exp = Experience(
            tokens=torch.tensor([0, 0], dtype=torch.long),
            prompt_length=1,
            action_mask=torch.tensor([False], dtype=torch.bool),
            logprobs=torch.tensor([0.0], dtype=torch.float),
            metrics={
                "success": 0.0,
                "reward": 0.0,
            },
        )
        exp.reward = 0.0
    return [exp]


def _get_jinja_env():
    """Initialize Jinja2 environment for template loading."""
    prompts_dir = Path(__file__).parent / "prompts"
    return Environment(
        loader=FileSystemLoader(str(prompts_dir)),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def format_qwen_math_prompt(prompt: str, attempt: int = 0, feedback: str = None) -> str:
    """
    Format prompt in Qwen-compatible format using \\boxed{} answer format.

    Args:
        prompt: The math problem prompt
        attempt: Current attempt number (0-based)
        feedback: Optional feedback from previous attempt

    Returns:
        Formatted prompt string with \\boxed{} reminder
    """
    if attempt == 0 or feedback is None:
        # First attempt - just the problem with boxed reminder
        return (
            f"{prompt}\n\n"
            "Please reason step by step, and put your final answer within \\boxed{}."
        )
    else:
        # Subsequent attempt - include feedback and boxed reminder
        return (
            f"Feedback: {feedback}\n\n"
            "Please try again. Please reason step by step, and put your final answer"
            " within \\boxed{}."
        )


def parse_boxed_response(response: str) -> Optional[str]:
    """Parse answer from \\boxed{} in response using find_boxed_answer."""
    return find_boxed_answer(response)


def my_math_verify(predicted_answer: str, ground_truth: str) -> bool:
    """
    Verify if the predicted math answer matches the ground truth using math_verify library.
    """
    if not predicted_answer or not ground_truth:
        return False

    try:
        # Parse and verify using math_verify library
        if "#" in ground_truth:
            # Handle cases like "### Answer"
            ground_truth = ground_truth.split("#")[-1].strip()
        gold = parse(ground_truth)
        answer = parse(predicted_answer)
        return verify(gold, answer)
    except Exception:
        # Fallback comparison
        return str(predicted_answer).strip().lower() == str(ground_truth).strip().lower()


def format_trajectory_for_reflection(trajectory: List[Dict[str, str]]) -> str:
    """
    Format trajectory for reflection analysis.
    Includes all messages including feedback.
    """
    formatted_lines = []
    step_counter = 0

    for msg in trajectory:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            formatted_lines.append(f"**System Prompt:**\n{content}\n" + "=" * 50)
        elif role == "user":
            formatted_lines.append(f"\n**Step {step_counter} - User:**")
            formatted_lines.append(f"{content}")
        elif role == "assistant":
            formatted_lines.append(f"\n**Step {step_counter} - Assistant Response:**")
            formatted_lines.append(f"{content}")
            step_counter += 1

    return "\n".join(formatted_lines)


def validate_reflect_report(report: Dict[str, Any], total_steps: int) -> Tuple[bool, bool]:
    """
    Validates the structure and content of the reflection report
    based on the reflection.j2 schema.

    Args:
        report: The reflection report to validate
        total_steps: Maximum number of steps in trajectory for retry_step bounds checking

    Returns:
        tuple[bool, bool]: (is_valid, is_perfect)
        - is_valid: Whether the report structure is valid
        - is_perfect: Whether the report indicates the trajectory is perfect (only meaningful if is_valid is True)
    """
    if (
        not isinstance(report, dict)
        or "trajectory_summary" not in report
        or "root_cause_analysis" not in report
        or "trajectory_outcome" not in report
    ):
        print("[Qwen Math Validation] Report is not a dict or missing keys.")
        return False, False

    outcome = report["trajectory_outcome"]

    if outcome == "success":
        # For success, we only need summary and no flaw analysis
        print("[Qwen Math Validation] success report validation successful.")
        return True, True

    elif outcome in ["success_but_inefficient", "failure"]:
        # For non-optimal outcomes, validate required fields
        improvement_suggestion = report.get("improvement_suggestion", None)
        retry_from_step = report.get("retry_from_step", None)

        if improvement_suggestion is None or retry_from_step is None:
            print(
                "[Qwen Math Validation] Missing 'improvement_suggestion' or 'retry_from_step'."
            )
            return False, False

        # check retry from step
        try:
            retry_from_step = int(retry_from_step)
        except (ValueError, TypeError):
            print(
                f"[Qwen Math Validation] 'retry_from_step' must be an integer. Got: {retry_from_step}"
            )
            return False, False
        if not isinstance(retry_from_step, int) or retry_from_step < 0:
            print(
                f"[Qwen Math Validation] 'retry_from_step' must be a non-negative integer. Got: {retry_from_step}"
            )
            return False, False
        # Check trajectory bounds if total_steps is provided
        if total_steps is not None:
            if retry_from_step >= total_steps:
                print(
                    f"[Qwen Math Validation] 'retry_from_step' ({retry_from_step}) exceeds trajectory bounds (0 to {total_steps - 1})."
                )
                return False, False
        print(f"[Qwen Math Validation] {outcome} report validation successful.")
        return True, False
    else:
        print(f"[Qwen Math Validation] Invalid trajectory_outcome: {outcome}")
        return False, False


def reflect_report_to_guidance_prompt(report: Dict[str, Any]) -> str:
    """
    Converts a validated reflection report into a structured, actionable
    guidance prompt for the agent's second attempt. This prompt is framed
    as an internal directive to ensure the model's output is clean for SFT.
    """
    # Convert the report dictionary to a formatted string
    report_str = json.dumps(report, indent=2, ensure_ascii=False)

    # Load and render template from qwen_math/prompts/
    jinja_env = _get_jinja_env()
    template = jinja_env.get_template("self_correction.j2")

    return template.render(report=report_str)


def create_experience_record(
    task_id: str,
    trajectory: List[Dict[str, str]],
    reward: float,
    success: bool,
    predicted_answer: str = "",
    ground_truth: str = "",
    attempt_type: str = "first",
    additional_metrics: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Create an experience record for data saving"""
    record = {
        "task_id": task_id,
        "timestamp": datetime.now().isoformat(),
        "attempt_type": attempt_type,
        "trajectory": trajectory,
        "reward": reward,
        "success": success,
        "predicted_answer": predicted_answer,
        "ground_truth": ground_truth,
    }

    if additional_metrics:
        record.update(additional_metrics)

    return record


def save_experience_data(task_id: str, experience_data: Dict[str, Any], data_dir: str):
    """Save experience data to file"""
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, f"{task_id}.json")

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(experience_data, f, ensure_ascii=False, indent=2)


def generate_default_experience() -> Experience:
    """Generate a default experience for failed cases"""
    return Experience(
        tokens=torch.tensor([0, 0], dtype=torch.long),
        prompt_length=1,
        action_mask=torch.tensor([False], dtype=torch.bool),
        logprobs=torch.tensor([0.0], dtype=torch.float),
        metrics={"success": 0.0, "reward": 0.0},
        reward=0.0,
    )
