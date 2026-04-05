import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from jinja2 import Environment, FileSystemLoader
import torch
from trinity.common.experience import Experience


def first_rollout(self, env) -> tuple[List[Dict[str, str]], float, bool, int, bool]:
    """Run a single rollout in Alfworld environment"""
    observation, info = env.reset()
    trajectory = []
    action_history = []  # Track all actions taken

    system_prompt = self.alfworld_system_template.render()
    trajectory.append({"role": "system", "content": system_prompt})

    default_reward = 0.0
    done = False
    reward = default_reward
    valid_format = True
    step = 0

    # Extract task description from the initial observation (only once at the beginning)
    task_description = extract_task_description(observation)

    for step in range(self.max_env_steps):
        # Extract admissible actions from info if available
        admissible_actions = info.get("admissible_commands", []) if isinstance(info, dict) else []

        trajectory.append(
            {"role": "user", "content": format_observation(
                current_observation=observation,
                task_description=task_description,
                current_step=step,
                action_history=action_history,
                admissible_actions=admissible_actions
            )}
        )

        # Get model response
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

        # Parse the response components
        think, action, error_msg = parse_response(response_text)
        if error_msg is not None:
            valid_format = False
            observation = f"{error_msg}"
            # 对于reward, done, info则保持默认值或者上一次的值
            trajectory.append({"role": "user", "content": f"Feedback: {error_msg}"})
            return trajectory, default_reward, False, step + 1, valid_format
        else:
            # Execute action in environment
            observation, reward, done, info = env.step(action)
            if action not in admissible_actions:
                valid_format = False
                observation = f"Invalid action '{action}' not in admissible actions."
                trajectory.append({"role": "user", "content": f"Feedback: {observation}"})
                return trajectory, default_reward, False, step + 1, valid_format

            # Track successfully executed actions for history
            if valid_format:
                action_history.append(action)

        # Check for consecutive action repetition (last 3 actions)
        if len(action_history) >= 3 and all(
                a == action_history[-1] for a in action_history[-3:]
        ):
            repeated_action = action_history[-1]
            feedback = f"Repeated invalid action {repeated_action} multiple times, task failed"
            trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
            valid_format = False
            return trajectory, default_reward, False, step + 1, valid_format

        if done:
            break

    # Generate feedback
    if reward >= 1.0 and step + 1 < self.max_env_steps:
        feedback = f"Task completed successfully (reward: {reward}/1.0), and satisfying the step limit ({step + 1}/{self.max_env_steps} steps)"
    elif reward >= 1.0 and step + 1 >= self.max_env_steps:
        feedback = (
            f"Task completed successfully (reward: {reward}/1.0), but exceeded the step limit ({step + 1}/{self.max_env_steps} steps)"
        )
    elif reward < 1.0 and step + 1 < self.max_env_steps:
        feedback = (
            f"Task not completed (reward: {reward}/1.0), but within the step limit ({step + 1}/{self.max_env_steps} steps)"
        )
    else:
        feedback = (
            f"Task not completed (reward: {reward}/1.0), and exceeded the step limit ({step + 1}/{self.max_env_steps} steps)"
        )

    # Add feedback to trajectory
    trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
    return trajectory, reward, False, step + 1, valid_format


def second_rollout(
        self,
        env,
        guidance_prompt: str,
        first_trajectory: List[Dict[str, str]],
        retry_step: int = 0,
) -> tuple[List[Dict[str, str]], List[Dict[str, str]], float, bool, int, bool]:
    """
    Performs rollout starting from a specific retry step, reusing previous responses.

    Args:
        env: The environment instance.
        guidance_prompt: The pre-generated guidance from reflection.
        first_trajectory: The full log of the initial attempt.
        retry_step: The step to start retry from (0-based, 0 means from beginning).

    Returns:
        A tuple containing (distill_trajectory, second_trajectory, reward, done status,
        step count, and format validity).
    """

    # Reset environment to start fresh
    observation, info = env.reset()
    trajectory = []
    distill_trajectory = []
    action_history = []  # Track all actions taken

    # Prepare system prompts
    original_system_prompt = self.alfworld_system_template.render()

    default_reward = 0.0
    done = False
    reward = default_reward
    valid_format = True

    # Extract task description from the initial observation (only once at the beginning)
    task_description = extract_task_description(observation)

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
                    think, action, _ = parse_response(msg["content"])
                    if think is not None and action is not None:
                        observation, reward, done, info = env.step(action)
                        action_history.append(action)
                    first_step += 1
                    step = first_step

                    if done:
                        # If environment finished during replay, no need to continue
                        return distill_trajectory, trajectory, reward, done, step, valid_format
                else:
                    break

        # Add guidance prompt as a separate system message before retry point
        guidance_system_msg = {"role": "system",
                               "content": f"# Previous Attempt Analysis & Guidance\n{guidance_prompt}"}
        trajectory.append(guidance_system_msg)
        # Don't add guidance to distill_trajectory to keep it clean

    else:
        # Starting from beginning - add system prompt with guidance
        merged_system_prompt = f"{original_system_prompt}\n\n# Previous Attempt Analysis & Guidance\n{guidance_prompt}"
        trajectory.append({"role": "system", "content": merged_system_prompt})
        distill_trajectory.append({"role": "system", "content": original_system_prompt})

    for step in range(step, self.max_env_steps):
        # Extract admissible actions from info if available
        admissible_actions = info.get("admissible_commands", []) if isinstance(info, dict) else []

        formatted_obs = format_observation(
            current_observation=observation,
            task_description=task_description,
            current_step=step,
            action_history=action_history,
            admissible_actions=admissible_actions
        )

        trajectory.append({"role": "user", "content": formatted_obs})
        distill_trajectory.append({"role": "user", "content": formatted_obs})

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
        think, action, error_msg = parse_response(response_text)
        if error_msg is not None:
            valid_format = False
            observation = f"{error_msg}"
        else:
            # Execute action in environment
            observation, reward, done, info = env.step(action)
            if action not in admissible_actions:
                valid_format = False
                observation = f"Invalid action '{action}' not in admissible actions."

            # Track successfully executed actions for history
            if valid_format:
                action_history.append(action)

        # Check for consecutive action repetition (last 3 actions)
        if len(action_history) >= 3 and all(
                a == action_history[-1] for a in action_history[-3:]
        ):
            repeated_action = action_history[-1]
            feedback = f"Repeated invalid action {repeated_action} multiple times, task failed"
            trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
            distill_trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
            valid_format = False
            return distill_trajectory, trajectory, default_reward, False, step + 1, valid_format

        if done:
            break

    print(f"[Second Rollout] - reward: {reward}, steps: {step + 1}")
    # Generate feedback
    if reward >= 1.0 and step + 1 < self.max_env_steps:
        feedback = f"Task completed successfully (reward: {reward}/1.0), and satisfying the step limit ({step + 1}/{self.max_env_steps} steps)"
    elif reward >= 1.0 and step + 1 >= self.max_env_steps:
        feedback = (
            f"Task completed successfully (reward: {reward}/1.0), but exceeded the step limit ({step + 1}/{self.max_env_steps} steps)"
        )
    elif reward < 1.0 and step + 1 < self.max_env_steps:
        feedback = (
            f"Task not completed (reward: {reward}/1.0), but within the step limit ({step + 1}/{self.max_env_steps} steps)"
        )
    else:
        feedback = (
            f"Task not completed (reward: {reward}/1.0), and exceeded the step limit ({step + 1}/{self.max_env_steps} steps)"
        )

    # Add feedback to trajectory
    trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})
    distill_trajectory.append({"role": "user", "content": f"Feedback: {feedback}"})

    return distill_trajectory, trajectory, reward, False, step + 1, valid_format


def eval_alfworld(self) -> List[Experience]:
    """Evaluate a single alfworld trajectory"""
    try:
        env = create_alfworld_environment(self.game_file_path)
        trajectory, reward, done, steps, valid_format = first_rollout(
            self, env
        )
        exp = self.model.convert_messages_to_experience(trajectory[:-1])
        exp.reward = reward
        exp.metrics = {
            "success": 1.0 if reward >= 1.0 else 0.0,
            "steps": steps,
            "reward": reward,
        }
        print(f"[Eval] First rollout - reward: {reward}, steps: {steps}")

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


def extract_task_description(observation: str) -> str:
    """
    Extract task description from the initial observation.
    The task description is typically in the format: "Your task is to: ..."

    Args:
        observation: Initial observation from environment

    Returns:
        Extracted task description string
    """
    # Look for pattern "Your task is to: <task>"
    match = re.search(r"Your task is to:\s*(.+?)(?:\n|$)", observation, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Fallback: return a portion of the observation
    return observation.split('\n')[-1] if '\n' in observation else observation


def format_observation(
        current_observation: str,
        task_description: str = "",
        current_step: int = 0,
        action_history: List[str] = None,
        admissible_actions: List[str] = None,
        history_length: int = 4
):
    """
    Format observation string with task context and limited action history.

    Args:
        current_observation: Current observation from environment
        task_description: Description of the task to complete
        current_step: Current step number
        action_history: List of all previous actions taken
        admissible_actions: List of currently admissible actions
        history_length: Maximum number of recent actions to display (default: 4)
    """
    if action_history is None:
        action_history = []
    if admissible_actions is None:
        admissible_actions = []

    # Check if this is the first step (no history)
    if current_step == 0 or not action_history:
        # First step - no history version
        prompt = f"""You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: {admissible_actions}.

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.

Format: <think>your reasoning process</think> <action>your chosen action</action>"""
    else:
        # Limit action history to most recent history_length items
        recent_actions = action_history[-history_length:] if len(action_history) > history_length else action_history

        # Format action history as a structured list
        action_history_str = "\n".join([f"  Step {current_step - len(recent_actions) + i}: {action}"
                                        for i, action in enumerate(recent_actions)])

        # Create formatted prompt with limited history
        prompt = f"""You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {len(action_history)} step(s). Below are the most recent {len(recent_actions)} actions you took:
{action_history_str}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: {admissible_actions}.

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.

Format: <think>your reasoning process</think> <action>your chosen action</action>"""

    return prompt


def parse_response(response):
    """
    Parse think and action components from response.
    Returns (think, action, error_message) tuple.
    - If successful: (think_content, action_content, None)
    - If error: (None, None, error_message)
    """
    try:
        # Use regex to extract all think and action components
        think_pattern = r"<think>\s*(.*?)\s*</think>"
        action_pattern = r"<action>\s*(.*?)\s*</action>"

        think_matches = re.findall(think_pattern, response, re.DOTALL)
        action_matches = re.findall(action_pattern, response, re.DOTALL)

        # Check for multiple think tags
        if len(think_matches) > 1:
            return None, None, f"Multiple <think> tags found ({len(think_matches)}). Only one <think></think> pair is allowed."

        # Check for multiple action tags
        if len(action_matches) > 1:
            return None, None, f"Multiple <action> tags found ({len(action_matches)}). Only one <action></action> pair is allowed."

        # Check if tags are missing
        if len(think_matches) == 0 and len(action_matches) == 0:
            return None, None, "Invalid response format, missing valid <think> or <action> tags, please ensure to follow the output format strictly: <think>...</think> <action>...</action>"
        elif len(think_matches) == 0:
            return None, None, "Invalid response format, missing valid <think> tag, please ensure to follow the output format strictly: <think>...</think> <action>...</action>"
        elif len(action_matches) == 0:
            return None, None, "Invalid response format, missing valid <action> tag, please ensure to follow the output format strictly: <think>...</think> <action>...</action>"

        think = think_matches[0].strip()
        action = action_matches[0].strip()

        return think, action, None
    except Exception:
        return None, None, "Unexpected error occurred while parsing response format."


def create_alfworld_environment(game_file, max_episode_steps=25):
    """
    Create alfworld environment

    Args:
        game_file: Path to the game file
        max_episode_steps: Maximum number of steps per episode (default: 50)
    """
    try:
        import textworld
        import textworld.gym
        from alfworld.agents.environment.alfred_tw_env import (
            AlfredDemangler,
            AlfredExpert,
            AlfredExpertType,
        )

        expert = AlfredExpert(expert_type=AlfredExpertType.HANDCODED)
        request_infos = textworld.EnvInfos(
            description=True, inventory=True, admissible_commands=True
        )

        env_id = textworld.gym.register_game(
            game_file, request_infos,
            max_episode_steps=max_episode_steps,
            asynchronous=True,
            wrappers=[AlfredDemangler(), expert]
        )
        env = textworld.gym.make(env_id)

        return env
    except ImportError as e:
        raise ImportError(
            f"Failed to import alfworld dependencies: {e}. "
            "Please install alfworld following the instructions at https://github.com/alfworld/alfworld"
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
    based on the new reflection.j2 schema.

    Args:
        report: The reflection report to validate
        total_steps: Total number of steps in trajectory for retry_step bounds checking

    Returns:
        Tuple[bool, bool]: (is_valid, is_perfect)
        - is_valid: Whether the report structure is valid
        - is_perfect: Whether the report indicates the trajectory is perfect (only meaningful if is_valid is True)
    """
    if (
            not isinstance(report, dict)
            or "trajectory_summary" not in report
            or "root_cause_analysis" not in report
            or "trajectory_outcome" not in report
    ):
        print("[R3L Alfworld Validation] Validation failed: Report is not a dict or missing keys.")
        return False, False

    outcome = report["trajectory_outcome"]
    analysis = report["root_cause_analysis"]

    if outcome == "success":
        # For OPTIMAL, we only need summary and no flaw analysis
        print("[R3L Alfworld Validation] success report validation successful.")
        return True, True

    elif outcome in ["success_but_inefficient", "failure"]:
        # For non-optimal outcomes, validate flaw_analysis structure
        improvement_suggestion = report.get("improvement_suggestion", None)
        retry_from_step = report.get("retry_from_step", None)

        if improvement_suggestion is None or retry_from_step is None:
            print("[R3L Alfworld Validation] Validation failed: Missing 'improvement_suggestion' or 'retry_from_step'.")
            return False, False

        # check retry from step
        try:
            retry_from_step = int(retry_from_step)
        except (ValueError, TypeError):
            print(f"[R3L Alfworld Validation] Validation failed: 'retry_from_step' must be an integer. Got: {retry_from_step}")
            return False, False
        if not isinstance(retry_from_step, int) or retry_from_step < 0:
            print(f"[R3L Alfworld Validation] Validation failed: 'retry_from_step' must be a non-negative integer. Got: {retry_from_step}")
            return False, False
        # Check trajectory bounds
        if retry_from_step >= total_steps:
            print(
                f"[R3L Alfworld Validation] Validation failed: 'retry_from_step' ({retry_from_step}) exceeds trajectory bounds (0 to {total_steps - 1}).")
            return False, False
        print(f"[R3L Alfworld Validation] {outcome} report validation successful.")
        return True, False
    else:
        print(f"[R3L Alfworld Validation] Invalid trajectory_outcome: {outcome}")
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
