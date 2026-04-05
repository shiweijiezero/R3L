# -*- coding: utf-8 -*-
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from jinja2 import Environment, FileSystemLoader

from trinity.common.experience import Experience
from trinity.utils.eval_utils import evaluate_equation, validate_equation


def first_rollout(self) -> tuple[List[Dict[str, str]], float, bool, str, str, int]:
    """Run countdown problem solving with multiple attempts (max 3 attempts) using multi-round interaction"""
    trajectory = []
    attempt_history = []  # Track attempt history for limited history display

    final_reward = 0.0
    final_success = False
    final_predicted_answer = ""
    attempt_count = 0

    # Try up to 3 attempts
    for attempt in range(self.max_attempts):
        attempt_count = attempt + 1

        # Format user prompt with history (limited to history_length)
        user_prompt = format_countdown_prompt(
            numbers=self.numbers,
            target=self.target,
            current_step=attempt,
            attempt_history=attempt_history,
            history_length=getattr(self, 'history_length', 4)
        )
        trajectory.append({"role": "user", "content": user_prompt})

        # Get model response
        responses = self.model.chat(
            trajectory,
            n=1,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Check if tokens exceed limit
        if responses[0].tokens.shape[0] >= 20480 - 4096:
            # 由于 chat 内部 tokenizer 会做截断，所以只要>= 最长限制 就直接终止
            return trajectory, final_reward, final_success, final_predicted_answer, str(self.target), attempt_count

        response_text = responses[0].response_text.strip()
        trajectory.append({"role": "assistant", "content": response_text})

        # Parse think and answer
        think, predicted_answer = parse_response(response_text)

        if think is None or predicted_answer is None:
            # Invalid format
            feedback = "Invalid response format. Please ensure you provide both <think>...</think> and <answer>...</answer> tags."
            trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
            # Record this failed attempt in history
            attempt_history.append({
                "equation": "Invalid format",
                "feedback": feedback
            })
            continue

        # Verify answer
        is_correct = countdown_verify(predicted_answer, self.numbers, self.target)

        if is_correct:
            final_reward = 1.0
            final_success = True
            final_predicted_answer = predicted_answer
            feedback = f"Correct! Your equation {predicted_answer} successfully equals {self.target} using the numbers {self.numbers}."
            trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
            break
        else:
            # Wrong answer
            if attempt < self.max_attempts - 1:
                feedback = f"Incorrect. Your equation {predicted_answer} does not work. Please try again."
                trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
            else:
                # Last attempt
                feedback = f"Incorrect. Your equation {predicted_answer} does not match the target {self.target}. Maximum attempts reached."
                trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
            final_predicted_answer = predicted_answer

            # Record this failed attempt in history
            attempt_history.append({
                "equation": predicted_answer,
                "feedback": feedback
            })

    return trajectory, final_reward, final_success, final_predicted_answer, str(self.target), attempt_count


def second_rollout(
        self,
        guidance_prompt: str,
        first_trajectory: List[Dict[str, str]],
        retry_step: int = 0,
) -> tuple[List[Dict[str, str]], List[Dict[str, str]], float, bool, str, str, int]:
    """
    Performs rollout with guidance from reflection.
    For countdown problems, we typically start from the beginning with guidance.
    """
    trajectory = []
    distill_trajectory = []
    attempt_history = []  # Track attempt history for limited history display

    final_reward = 0.0
    final_success = False
    final_predicted_answer = ""
    attempt_count = 0

    # Try up to 3 attempts
    for attempt in range(self.max_attempts):
        attempt_count = attempt + 1

        # Format user prompt with history and guidance
        if attempt == 0:
            # First attempt includes guidance
            user_prompt = format_countdown_prompt_with_guidance(
                numbers=self.numbers,
                target=self.target,
                current_step=attempt,
                attempt_history=attempt_history,
                guidance_prompt=guidance_prompt,
                history_length=getattr(self, 'history_length', 4)
            )
            # For distill trajectory, use prompt without guidance
            distill_user_prompt = format_countdown_prompt(
                numbers=self.numbers,
                target=self.target,
                current_step=attempt,
                attempt_history=attempt_history,
                history_length=getattr(self, 'history_length', 4)
            )
        else:
            # Subsequent attempts don't repeat guidance
            user_prompt = format_countdown_prompt(
                numbers=self.numbers,
                target=self.target,
                current_step=attempt,
                attempt_history=attempt_history,
                history_length=getattr(self, 'history_length', 4)
            )
            distill_user_prompt = user_prompt

        trajectory.append({"role": "user", "content": user_prompt})
        distill_trajectory.append({"role": "user", "content": distill_user_prompt})

        # Get model response with guidance
        responses = self.model.chat(
            trajectory,
            n=1,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Check if tokens exceed limit
        if responses[0].tokens.shape[0] >= 20480 - 4096:
            # 由于 chat 内部 tokenizer 会做截断，所以只要>= 最长限制 就直接终止
            return distill_trajectory, trajectory, final_reward, final_success, final_predicted_answer, str(self.target), attempt_count

        response_text = responses[0].response_text.strip()
        trajectory.append({"role": "assistant", "content": response_text})
        distill_trajectory.append({"role": "assistant", "content": response_text})

        # Parse think and answer
        think, predicted_answer = parse_response(response_text)

        if think is None or predicted_answer is None:
            # Invalid format
            feedback = "Invalid response format. Please ensure you provide both <think>...</think> and <answer>...</answer> tags."
            trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
            distill_trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
            # Record this failed attempt in history
            attempt_history.append({
                "equation": "Invalid format",
                "feedback": feedback
            })
            continue

        # Verify answer
        is_correct = countdown_verify(predicted_answer, self.numbers, self.target)

        if is_correct:
            final_reward = 1.0
            final_success = True
            final_predicted_answer = predicted_answer
            feedback = f"Correct! Your equation {predicted_answer} successfully equals {self.target}."
            trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
            distill_trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
            break
        else:
            # Wrong answer
            if attempt < self.max_attempts - 1:
                feedback = f"Incorrect. Your equation {predicted_answer} does not work. Please try again."
                trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
                distill_trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
            else:
                # Last attempt
                feedback = f"Incorrect. Your equation {predicted_answer} does not match the target {self.target}. Maximum attempts reached."
                trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
                distill_trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
            final_predicted_answer = predicted_answer

            # Record this failed attempt in history
            attempt_history.append({
                "equation": predicted_answer,
                "feedback": feedback
            })

    return distill_trajectory, trajectory, final_reward, final_success, final_predicted_answer, str(self.target), attempt_count


def eval_countdown(self) -> List[Experience]:
    """Evaluate a single countdown problem"""
    print("[R3L Countdown Eval] Starting evaluation...")
    try:
        trajectory, reward, success, predicted_answer, ground_truth, attempts = first_rollout(self)
        exp = self.model.convert_messages_to_experience(trajectory[:-1])
        exp.reward = reward
        exp.metrics = {
            "success": 1.0 if success else 0.0,
            "reward": reward,
            "attempts": attempts,
        }
        print(f"[R3L Countdown Eval] Completed - Reward: {reward}, Success: {success}, Attempts: {attempts}")

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
                attempt_type="evaluation"
            )
            save_experience_data(
                task_id=f"{eval_task_id}_eval",
                experience_data=eval_record,
                data_dir=self.eval_dir
            )
    except Exception as e:
        print(f"[R3L Countdown Eval] Evaluation failed - Error: {str(e)}")
        exp = Experience(
            tokens=torch.tensor([0, 0], dtype=torch.long),
            prompt_length=1,
            action_mask=torch.tensor([False], dtype=torch.bool),
            logprobs=torch.tensor([0.0], dtype=torch.float),
            metrics={
                "success": 0.0,
                "reward": 0.0,
            }
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


def parse_response(response: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse think and answer from countdown response"""
    try:
        # Extract think section
        think_pattern = r"<think>\s*(.*?)\s*</think>"
        think_match = re.search(think_pattern, response, re.DOTALL)
        think = think_match.group(1).strip() if think_match else None

        # Extract answer from <answer> tags
        answer_pattern = r"<answer>\s*(.*?)\s*</answer>"
        answer_match = re.search(answer_pattern, response, re.DOTALL | re.IGNORECASE)
        if answer_match:
            answer = answer_match.group(1).strip()
        else:
            answer = None

        return think, answer
    except Exception as e:
        print(f"Error parsing response: {e}")
        return None, None


def countdown_verify(predicted_answer: str, numbers: List[int], target: int) -> bool:
    """
    Verify if the predicted countdown equation is correct.
    """
    if not predicted_answer:
        print("Predicted answer is empty.")
        return False

    # Extract equation from predicted answer
    equation = predicted_answer

    # Validate equation uses correct numbers
    if not validate_equation(equation, numbers):
        print("Equation validation failed: uses invalid numbers.")
        return False

    # Evaluate equation
    try:
        result = evaluate_equation(equation)
        if result is None:
            print("Equation evaluation returned None.")
            return False

        if abs(result - target) < 1e-5:  # Account for floating point precision
            print(f"Equation evaluation successful: matches target, {result}, {target}.")
            return True
        else:
            print(f"Equation evaluation result {result} does not match target {target}.")
            return False
    except Exception as e:
        print(f"Error evaluating equation: {e}")
        return False


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
    based on the alfworld reflection.j2 schema.

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
        print("[R3L Countdown Validation] Report is not a dict or missing keys.")
        return False, False

    outcome = report["trajectory_outcome"]

    if outcome == "success":
        # For success, we only need summary and no flaw analysis
        print("[R3L Countdown Validation] success report validation successful.")
        return True, True

    elif outcome in ["success_but_inefficient", "failure"]:
        # For non-optimal outcomes, validate required fields
        improvement_suggestion = report.get("improvement_suggestion", None)
        retry_from_step = report.get("retry_from_step", None)

        if improvement_suggestion is None or retry_from_step is None:
            print("[R3L Countdown Validation] Missing 'improvement_suggestion' or 'retry_from_step'.")
            return False, False

        # check retry from step
        try:
            retry_from_step = int(retry_from_step)
        except (ValueError, TypeError):
            print(f"[R3L Countdown Validation] 'retry_from_step' must be an integer. Got: {retry_from_step}")
            return False, False
        if not isinstance(retry_from_step, int) or retry_from_step < 0:
            print(f"[R3L Countdown Validation] 'retry_from_step' must be a non-negative integer. Got: {retry_from_step}")
            return False, False
        # Check trajectory bounds if total_steps is provided
        if total_steps is not None:
            if retry_from_step >= total_steps:
                print(
                    f"[R3L Countdown Validation] 'retry_from_step' ({retry_from_step}) exceeds trajectory bounds (0 to {total_steps - 1}).")
                return False, False
        print(f"[R3L Countdown Validation] {outcome} report validation successful.")
        return True, False
    else:
        print(f"[R3L Countdown Validation] Invalid trajectory_outcome: {outcome}")
        return False, False


def reflect_report_to_guidance_prompt(report: Dict[str, Any]) -> str:
    """
    Converts a validated reflection report into a structured, actionable
    guidance prompt for the agent's second attempt. This prompt is framed
    as an internal directive to ensure the model's output is clean for SFT.
    """
    # Convert the report dictionary to a formatted string
    report_str = json.dumps(report, indent=2, ensure_ascii=False)

    # Load and render template
    jinja_env = _get_jinja_env()
    template = jinja_env.get_template("self_correction.j2")

    return template.render(report=report_str)


def save_experience_data(
        task_id: str,
        experience_data: Dict,
        data_dir: str
) -> str:
    """
    Save experience data including trajectory, rewards, and attempts to a JSON file.

    Args:
        task_id: Unique identifier for the task
        experience_data: Dictionary containing experience information
        data_dir: Directory to save the data

    Returns:
        Path to the saved file
    """
    os.makedirs(data_dir, exist_ok=True)
    filename = f"{task_id}.json"
    filepath = os.path.join(data_dir, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(experience_data, f, indent=2, ensure_ascii=False)

    return filepath


def create_experience_record(
        task_id: str,
        trajectory: List[Dict[str, str]],
        reward: float,
        success: bool,
        predicted_answer: str,
        ground_truth: str,
        attempt_type: str,
        additional_metrics: Optional[Dict] = None
) -> Dict:
    """
    Create a structured experience record.

    Args:
        task_id: Task identifier
        trajectory: Conversation trajectory
        reward: Final reward
        success: Whether the task was successful
        predicted_answer: Model's predicted answer
        ground_truth: Correct answer
        attempt_type: Type of attempt (e.g., 'first', 'second', 'evaluation')
        additional_metrics: Optional additional metrics

    Returns:
        Experience record dictionary
    """
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
        record["additional_metrics"] = additional_metrics

    return record


def format_countdown_prompt(
        numbers: List[int],
        target: int,
        current_step: int,
        attempt_history: List[Dict[str, str]],
        history_length: int = 4
) -> str:
    """
    Format countdown prompt with limited history.

    Args:
        numbers: Available numbers for the countdown problem
        target: Target number to achieve
        current_step: Current attempt number
        attempt_history: List of previous attempts with equations and feedback
        history_length: Maximum number of previous attempts to show (default: 4)

    Returns:
        Formatted prompt string
    """
    if current_step == 0 or not attempt_history:
        # First attempt - no history
        prompt = f"""You are an expert at solving countdown number problems.
Your current task is: Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once.

Now it's your turn to solve this problem.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should provide your equation answer and present it within <answer> </answer> tags, for example <answer>(1 + 2) / 3</answer>."""
    else:
        # Show limited history
        recent_attempts = attempt_history[-history_length:] if len(attempt_history) > history_length else attempt_history

        # Format attempt history as a list
        history_lines = []
        for idx, attempt in enumerate(recent_attempts):
            attempt_num = current_step - len(recent_attempts) + idx + 1
            history_lines.append(f"  Attempt {attempt_num}: {attempt['equation']} -> {attempt['feedback']}")

        attempt_history_str = "\n".join(history_lines)

        prompt = f"""You are an expert at solving countdown number problems.
Your task is: Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once.
Prior to this attempt, you have already made {current_step} attempt(s). Below are the most recent {len(recent_attempts)} attempts and their feedback:
{attempt_history_str}
You are now at attempt {current_step + 1}.

Now it's your turn to solve this problem.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should provide your equation answer and present it within <answer> </answer> tags, for example <answer>(1 + 2) / 3</answer>."""

    return prompt


def format_countdown_prompt_with_guidance(
        numbers: List[int],
        target: int,
        current_step: int,
        attempt_history: List[Dict[str, str]],
        guidance_prompt: str,
        history_length: int = 4
) -> str:
    """
    Format countdown prompt with limited history and guidance from reflection.

    Args:
        numbers: Available numbers for the countdown problem
        target: Target number to achieve
        current_step: Current attempt number
        attempt_history: List of previous attempts with equations and feedback
        guidance_prompt: Guidance from reflection analysis
        history_length: Maximum number of previous attempts to show (default: 4)

    Returns:
        Formatted prompt string with guidance
    """
    base_prompt = format_countdown_prompt(numbers, target, current_step, attempt_history, history_length)

    # Insert guidance before the final instruction
    split_marker = "Now it's your turn"
    base_parts = base_prompt.split(split_marker)
    base_prefix = base_parts[0] if len(base_parts) > 0 else base_prompt

    prompt_with_guidance = f"""{base_prefix}
# Previous Attempt Analysis & Guidance
{guidance_prompt}

Now it's your turn to solve this problem.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should provide your equation answer and present it within <answer> </answer> tags, for example <answer>(1 + 2) / 3</answer>."""

    return prompt_with_guidance
