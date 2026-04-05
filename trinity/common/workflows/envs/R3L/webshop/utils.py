import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jinja2 import Environment, FileSystemLoader
import torch
from trinity.common.experience import Experience


def first_rollout(self, env, session_id) -> tuple[List[Dict[str, str]], float, bool, int, bool]:
    """Run a single rollout"""
    # print(f"About to reset env with session_id: {session_id}")
    env.reset(session=session_id)
    observation = env.observation
    trajectory = []
    action_history = []  # Track last 3 actions for repetition detection

    system_prompt = self.webshop_system_template.render()
    trajectory.append({"role": "system", "content": system_prompt})

    default_reward = -0.1
    reward = default_reward
    valid_format = True
    step = 0

    for step in range(self.max_env_steps):
        available_actions = env.get_available_actions()
        trajectory.append(
            {"role": "user", "content": format_observation(observation, available_actions)}
        )

        # Get model response with experience guidance
        responses = self.model.chat(
            trajectory,
            n=1,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Check if tokens exceed limit
        if responses[0].tokens.shape[0] >= 20480 - 512:
            # 由于 chat 内部 tokenizer 会做截断，所以只要>= 最长限制 就直接终止
            return trajectory, default_reward, False, step + 1, False

        response_text = responses[0].response_text.strip()
        trajectory.append({"role": "assistant", "content": response_text})

        # Parse the three components for action execution
        think, action = parse_response(response_text)
        if action is None:
            valid_format = False
            feedback = "Invalid response format, missing valid <think> or <action> tags, please ensure to follow the output format strictly: <think>...</think> <action>...</action>"
            trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
            # print(f"Terminating due to invalid response format: {response_text}")
            return trajectory, default_reward, False, step + 1, valid_format

        # Check for consecutive action repetition
        action_history.append(action)
        if len(action_history) > 2:
            action_history.pop(0)

        # If last 2 actions are the same, terminate with failure
        if len(action_history) >= 2 and all(
                action == action_history[0] for action in action_history
        ) and "next" not in action.lower() and "prev" not in action.lower() and "search" not in action.lower():
            feedback = f"Repeated invalid action {action} multiple times, shopping task failed"
            trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
            # print(f"Terminating due to 5 consecutive identical actions: {action_text}")
            valid_format = False
            return trajectory, default_reward, False, step + 1, valid_format

        # Validate and execute action in environment
        action_valid, error_msg = validate_action(action, available_actions)
        if action_valid:
            observation, reward, done, info = env.step(action)
        else:
            observation, reward, done = error_msg, default_reward, False

        if done:
            break

    # Generate timeout feedback
    if reward >= 1.0 and step + 1 < self.max_env_steps:
        feedback = f"Shopping task completed successfully (reward: {reward}/1.0), and satisfying the step limit ({step + 1}/{self.max_env_steps} steps)"
    elif reward >= 1.0 and step + 1 >= self.max_env_steps:
        feedback = (
            f"Shopping task completed successfully (reward: {reward}/1.0), but exceeded the step limit ({step + 1}/{self.max_env_steps} steps)"
        )
    elif reward < 1.0 and step + 1 < self.max_env_steps:
        feedback = (
            f"Shopping task not completed (reward: {reward}/1.0), but within the step limit ({step + 1}/{self.max_env_steps} steps). It may not satisfy the Attribute Matching, Option Matching, or Price Matching requirements, please you carefully check and ensure all requirements are satisfied."
        )
    else:
        feedback = (
            f"Shopping task not completed (reward: {reward}/1.0), and exceeded the step limit ({step + 1}/{self.max_env_steps} steps). It may not satisfy the Attribute Matching, Option Matching, or Price Matching requirements, please you carefully check and ensure all requirements are satisfied."
        )

    # Add timeout feedback to trajectory
    trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
    return trajectory, reward, False, step + 1, valid_format

def second_rollout(
        self,
        env,
        session_id: int,
        guidance_prompt: str,
        first_trajectory: List[Dict[str, str]],
        retry_step: int = 0,
) -> tuple[List[Dict[str, str]], List[Dict[str, str]], float, bool, int, bool]:
    """
    Performs rollout starting from a specific retry step, reusing previous responses.

    Args:
        env: The environment instance.
        session_id: The ID for the current task session.
        guidance_prompt: The pre-generated guidance from reflection.
        first_trajectory: The full log of the initial attempt.
        retry_step: The step to start retry from (0-based, 0 means from beginning).

    Returns:
        A tuple containing (distill_trajectory, second_trajectory, reward, done status,
        step count, and format validity).
    """

    # Reset environment to start fresh
    env.reset(session=session_id)
    observation = env.observation
    trajectory = []
    distill_trajectory = []
    action_history = []  # Track last 3 actions for repetition detection

    # Prepare system prompts
    original_system_prompt = self.webshop_system_template.render()

    default_reward = -0.1
    reward = default_reward
    valid_format = True

    # Copy responses from first trajectory up to retry_step
    step = 0
    if retry_step > 0:
        # Add original system prompt only
        trajectory.append({"role": "system", "content": original_system_prompt})
        distill_trajectory.append({"role": "system", "content": original_system_prompt})

        # Replay first trajectory up to retry_step to restore environment state
        first_step = 0
        for msg in first_trajectory[1:]:  # Skip system message
            if msg["role"] == "user":
                # This is an observation - copy it and continue
                trajectory.append(msg)
                distill_trajectory.append(msg)
            elif msg["role"] == "assistant":
                if first_step < retry_step:
                    # Copy the assistant response from first trajectory
                    trajectory.append(msg)
                    distill_trajectory.append(msg)

                    # Execute the action to restore environment state
                    think, action = parse_response(msg["content"])
                    if think is not None and action is not None:
                        action_valid, error_msg = validate_action(action, env.get_available_actions())
                        if action_valid:
                            observation, reward, done, info = env.step(action)
                            action_history.append(action)
                            if len(action_history) > 2:
                                action_history.pop(0)
                        else:
                            # If action becomes invalid during replay, start from beginning
                            retry_step = 0
                            break
                    first_step += 1
                    step = first_step

                    if done:
                        # If environment finished during replay, no need to continue
                        return distill_trajectory, trajectory, reward, done, step, valid_format
                else:
                    break

        # Add guidance prompt as a separate system message before retry point
        guidance_system_msg = {"role": "system", "content": f"# Previous Attempt Analysis & Guidance\n{guidance_prompt}"}
        trajectory.append(guidance_system_msg)
        # Don't add guidance to distill_trajectory to keep it clean

    else:
        # Starting from beginning - add system prompt with guidance
        merged_system_prompt = f"{original_system_prompt}\n\n# Previous Attempt Analysis & Guidance\n{guidance_prompt}"
        trajectory.append({"role": "system", "content": merged_system_prompt})
        distill_trajectory.append({"role": "system", "content": original_system_prompt})

    for step in range(step, self.max_env_steps):
        available_actions = env.get_available_actions()
        trajectory.append(
            {"role": "user", "content": format_observation(observation, available_actions)}
        )
        distill_trajectory.append(
            {"role": "user", "content": format_observation(observation, available_actions)}
        )

        # Get model response with guidance
        responses = self.model.chat(
            trajectory,
            n=1,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Check if tokens exceed limit
        if responses[0].tokens.shape[0] >= 20480 - 512:
            # 由于 chat 内部 tokenizer 会做截断，所以只要>= 最长限制 就直接终止
            return distill_trajectory, trajectory, default_reward, False, step + 1, False

        response_text = responses[0].response_text.strip()
        trajectory.append({"role": "assistant", "content": response_text})
        distill_trajectory.append({"role": "assistant", "content": response_text})

        # Parse the response
        think, action = parse_response(response_text)
        if think is None or action is None:
            valid_format = False
            feedback = "Invalid response format, missing valid <think> or <action> tags, please ensure to follow the output format strictly: <think>...</think> <action>...</action>"
            trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
            distill_trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
            return distill_trajectory, trajectory, default_reward, False, step + 1, valid_format

        # Check for consecutive action repetition
        action_history.append(action)
        if len(action_history) > 2:
            action_history.pop(0)

        # If last 2 actions are the same, terminate with failure
        if len(action_history) >= 2 and all(
                action == action_history[0] for action in action_history
        ) and "next" not in action.lower() and "prev" not in action.lower() and "search" not in action.lower():
            feedback = f"Repeated invalid action {action} multiple times, shopping task failed"
            trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
            distill_trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
            valid_format = False
            return distill_trajectory, trajectory, default_reward, False, step + 1, valid_format

        # Validate and execute action in environment
        action_valid, error_msg = validate_action(action, available_actions)
        if action_valid:
            observation, reward, done, info = env.step(action)
        else:
            observation, reward, done = error_msg, default_reward, False

        if done:
            break

    # Generate feedback
    if reward >= 1.0 and step + 1 < self.max_env_steps:
        feedback = f"Shopping task completed successfully (reward: {reward}/1.0), and satisfying the step limit ({step + 1}/{self.max_env_steps} steps)"
    elif reward >= 1.0 and step + 1 >= self.max_env_steps:
        feedback = (
            f"Shopping task completed successfully (reward: {reward}/1.0), but exceeded the step limit ({step + 1}/{self.max_env_steps} steps)"
        )
    elif reward < 1.0 and step + 1 < self.max_env_steps:
        feedback = (
            f"Shopping task not completed (reward: {reward}/1.0), but within the step limit ({step + 1}/{self.max_env_steps} steps). It may not satisfy the Attribute Matching, Option Matching, or Price Matching requirements, please you carefully check and ensure all requirements are satisfied."
        )
    else:
        feedback = (
            f"Shopping task not completed (reward: {reward}/1.0), and exceeded the step limit ({step + 1}/{self.max_env_steps} steps). It may not satisfy the Attribute Matching, Option Matching, or Price Matching requirements, please you carefully check and ensure all requirements are satisfied."
        )

    # Add feedback to trajectory
    trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
    distill_trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})

    # For compatibility, return the same trajectory as both distill_trajectory and second_trajectory
    # since we're starting fresh instead of resuming from a checkpoint
    return distill_trajectory, trajectory, reward, False, step + 1, valid_format

def eval_webshop(self) -> List[Experience]:
    """Evaluate a single webshop trajectory"""
    try:
        trajectory, reward, done, steps, valid_format = first_rollout(
            self, self.env, self.session_id
        )
        exp = self.model.convert_messages_to_experience(trajectory[:-1])
        exp.reward = reward
        exp.metrics = {
            "success": 1.0 if reward >= 1.0 else 0.0,
            "steps": steps,
            "reward": reward,
        }
        print(f"[WebShop Eval] Rollout - reward: {reward}, steps: {steps}, valid_format: {valid_format}")

        if self.whether_save_data:
            # Save evaluation data
            eval_task_id = f"{str(self.task.batch_id).replace('/', '_')}_{self.task.task_id}"
            eval_record = create_experience_record(
                task_id=eval_task_id,
                trajectory=trajectory,
                reward=reward,
                steps=steps,
                success=reward >= 1.0,
                attempt_type="evaluation"
            )
            save_experience_data(
                task_id=f"{eval_task_id}_eval",
                experience_data=eval_record,
                data_dir=self.eval_dir
            )
    except Exception as e:
        # logger.warning(f"Single rollout failed during eval: {e}")
        task_id = f"{str(self.task.batch_id).replace('/', '_')}_{self.task.task_id}"
        exp = Experience(
            tokens=torch.tensor([0, 0], dtype=torch.long),
            prompt_length=1,
            action_mask=torch.tensor([False], dtype=torch.bool),
            logprobs=torch.tensor([0.0], dtype=torch.float),
            metrics={
                "success": 0.0,
                "reward": -0.1,
            }
        )
        exp.reward = -0.1
    return [exp]

def _get_jinja_env():
    """Initialize Jinja2 environment for template loading."""
    prompts_dir = Path(__file__).parent / "prompts"
    return Environment(
        loader=FileSystemLoader(str(prompts_dir)),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def format_observation(observation: str, available_actions: dict):
    """Format observation with format reminder for each turn"""
    formatted_prompt = f"""Environment Observation: {observation}
Available Actions: {available_actions}

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an action and present it within <action> </action> tags.

Format: <think>your reasoning process</think> <action>your chosen action</action>"""
    return formatted_prompt


def parse_response(response):
    """Parse all three components from response with a single regex"""
    think, action = None, None
    try:
        # Use single regex to extract all three components at once
        pattern = r"<think>\s*(.*?)\s*</think>.*?<action>\s*(.*?)\s*</action>"
        match = re.search(pattern, response, re.DOTALL)

        if match:
            think, action = match.group(1).strip(), match.group(2).strip()
    except Exception:
        pass
    return think, action


def validate_action(action, available_actions):
    """Validate action format and availability"""
    import re

    # Parse action format: action_name[action_arg]
    pattern = re.compile(r"(.+)\[(.+)\]")
    m = re.match(pattern, action)
    if m is None:
        return (
            False,
            "Invalid action format. You should use format: action_name[action_arg], like search[query] or click[button].",
        )

    action_name, action_arg = m.groups()
    action_name = action_name.strip()
    action_arg = action_arg.strip()

    # Validate search action
    if action_name == "search":
        if not action_arg:
            return (
                False,
                "Invalid search action, please type in the query you want to search in the square brackets.",
            )
        if not available_actions["has_search_bar"]:
            return (
                False,
                "Cannot perform search action without search bar. Please click the Back to Search button first.",
            )
        return True, ""

    # Validate click action
    elif action_name == "click":
        if not action_arg:
            return (
                False,
                "Invalid click action, please specify the button name in the square brackets.",
            )
        # Convert to lowercase for comparison (as clickables are typically lowercase)
        action_arg_lower = action_arg.lower()
        if action_arg_lower not in available_actions["clickables"]:
            return (
                False,
                f"Button '{action_arg}' not found on current page. Available buttons: {available_actions['clickables']}",
            )
        return True, ""

    # Unknown action
    else:
        return (
            False,
            f"Unknown action '{action_name}'. Only 'search' and 'click' actions are supported.",
        )


def format_trajectory_for_reflection(trajectory: List[Dict[str, str]]) -> str:
    """
    Correctly formats the trajectory for reflection, including the system prompt
    and numbering the user/assistant turns.
    """
    formatted_lines = []
    # 使用一个计数器来追踪 user/assistant 的交互轮次
    turn_counter = 0  # 从 0 开始计数

    for msg in trajectory:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            # 系统提示不计入步骤，但必须作为规则展示在最前面
            formatted_lines.append(f"**System Rules & Context:**\n{content}\n" + "=" * 30)
        elif role == "user":
            # 标记一个新回合的开始
            formatted_lines.append(f"\n**Step {turn_counter}**")
            formatted_lines.append(f"  - User Observation/Feedback:\n    {content.strip()}")
        elif role == "assistant":
            # 助理的思考和行动
            formatted_lines.append(f"  - Assistant Thought & Action:\n    {content.strip()}")
            # 一轮完整的 user-assistant 对话结束后，回合数增加
            turn_counter += 1

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
        print("[R3L WebShop Validation] Report is not a dict or missing keys.")
        return False, False

    outcome = report["trajectory_outcome"]

    if outcome == "success":
        # For success, we only need summary and no flaw analysis
        print("[R3L WebShop Validation] success report validation successful.")
        return True, True

    elif outcome in ["success_but_inefficient", "failure"]:
        # For non-optimal outcomes, validate required fields
        improvement_suggestion = report.get("improvement_suggestion", None)
        retry_from_step = report.get("retry_from_step", None)

        if improvement_suggestion is None or retry_from_step is None:
            print("[R3L WebShop Validation] Missing 'improvement_suggestion' or 'retry_from_step'.")
            return False, False

        # check retry from step
        try:
            retry_from_step = int(retry_from_step)
        except (ValueError, TypeError):
            print(f"[R3L WebShop Validation] 'retry_from_step' must be an integer. Got: {retry_from_step}")
            return False, False
        if not isinstance(retry_from_step, int) or retry_from_step < 0:
            print(f"[R3L WebShop Validation] 'retry_from_step' must be a non-negative integer. Got: {retry_from_step}")
            return False, False
        # Check trajectory bounds if total_steps is provided
        if total_steps is not None:
            if retry_from_step >= total_steps:
                print(
                    f"[R3L WebShop Validation] 'retry_from_step' ({retry_from_step}) exceeds trajectory bounds (0 to {total_steps - 1}).")
                return False, False
        print(f"[R3L WebShop Validation] {outcome} report validation successful.")
        return True, False
    else:
        print(f"[R3L WebShop Validation] Invalid trajectory_outcome: {outcome}")
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
    Save experience data including trajectory, rewards, and steps to a JSON file.

    Args:
        task_id: Unique identifier for the task
        experience_data: Dictionary containing experience information
        data_dir: Directory to save the data

    Returns:
        Path to the saved file
    """
    os.makedirs(data_dir, exist_ok=True)

    # Add timestamp for uniqueness
    filename = f"{task_id}.json"
    filepath = os.path.join(data_dir, filename)

    # Ensure experience_data is JSON serializable
    serializable_data = {}
    for key, value in experience_data.items():
        if isinstance(value, torch.Tensor):
            serializable_data[key] = value.tolist()
        elif hasattr(value, '__dict__'):
            # For complex objects, convert to dict representation
            serializable_data[key] = str(value)
        else:
            serializable_data[key] = value

    # Add metadata
    serializable_data["saved_at"] = datetime.now().isoformat()
    serializable_data["task_id"] = task_id

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        # print(f"Experience data saved to: {filepath}")
        return filepath
    except Exception as e:
        # print(f"Failed to save experience data: {e}")
        return ""


def create_experience_record(
        task_id: str,
        trajectory: List[Dict[str, str]],
        reward: float,
        steps: int,
        success: bool,
        attempt_type: str = "first",
        reflection_data: Optional[Dict] = None,
        additional_metrics: Optional[Dict] = None
) -> Dict:
    """
    Create a structured experience record for saving.

    Args:
        task_id: Unique identifier for the task
        trajectory: List of conversation messages
        reward: Final reward received
        steps: Number of steps taken
        success: Whether the task was completed successfully
        attempt_type: Type of attempt ("first", "second", "reflect")
        reflection_data: Optional reflection analysis data
        additional_metrics: Additional metrics to record

    Returns:
        Dictionary containing structured experience data
    """
    experience_record = {
        "task_id": task_id,
        "attempt_type": attempt_type,
        "trajectory": trajectory,
        "metrics": {
            "reward": reward,
            "steps": steps,
            "success": success,
            "trajectory_length": len(trajectory)
        },
        "created_at": datetime.now().isoformat()
    }

    if reflection_data:
        experience_record["reflection"] = reflection_data

    if additional_metrics:
        experience_record["metrics"].update(additional_metrics)

    return experience_record
