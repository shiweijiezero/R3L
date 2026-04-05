# -*- coding: utf-8 -*-
"""Step-level GRPO/OPMD/GSPO workflows for DAPO math with Qwen3 compatibility.

Qwen3's chat template removes thinking tokens from context between turns (dynamic context),
so multi-turn conversations cannot be concatenated as a single training sequence.
These workflows decompose each attempt into an independent Experience via
RewardPropagationWorkflow, enabling step-level training.

For DAPO, each "step" is a math problem solving attempt (max 3 attempts).
"""
import copy
from pathlib import Path
from typing import List, Optional

import torch
from jinja2 import Environment, FileSystemLoader

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.envs.R3L.dapo import utils
from trinity.common.workflows.step_wise_workflow import RewardPropagationWorkflow
from trinity.common.workflows.workflow import WORKFLOWS, Task


class StepLevelDapoBaseWorkflow(RewardPropagationWorkflow):
    """Base step-level workflow for DAPO mathematical problem solving.

    Each attempt at solving the math problem produces a separate Experience via
    model.extract_experience_from_history(). All attempts in a rollout share
    the same final reward (reward propagation).
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
            task=task,
            model=model,
            auxiliary_models=auxiliary_models,
            use_openai_client=False,
        )
        self.temperature = getattr(task.rollout_args, "temperature", 1.0)
        self.max_attempts = 3
        self.max_tokens = 4096
        self.task = task
        self.is_eval = task.is_eval
        self.n = task.repeat_times
        self.run_id_base = 0

        # Jinja2 templates
        prompts_dir = Path(__file__).parent / "prompts"
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(prompts_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.dapo_system_template = self.jinja_env.get_template("math_system.j2")

        # Default experience for error/empty cases
        self.default_exp = Experience(
            tokens=torch.tensor([0, 0], dtype=torch.long),
            prompt_length=1,
            action_mask=torch.tensor([False], dtype=torch.bool),
            logprobs=torch.tensor([0.0], dtype=torch.float),
            metrics={"success": 0.0, "reward": 0.0},
            reward=0.0,
        )

        # Per-rollout state
        self.done = False
        self.final_reward = 0.0
        self.predicted_answer = ""
        self.memory = []

        print(f"Initializing {self.__class__.__name__}, temperature={self.temperature}")
        self.reset(task)

    def reset(self, task: Task):
        self.is_eval = task.is_eval
        self.task = task
        self.n = task.repeat_times
        self.temperature = getattr(task.rollout_args, "temperature", 1.0)

        # Extract prompt and ground truth from task
        if hasattr(task, "raw_task") and task.raw_task:
            raw_task = task.raw_task

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
            elif "question" in raw_task and "answer" in raw_task:
                self.prompt = raw_task.get("question", "")
                self.ground_truth = raw_task.get("answer", "")
            else:
                self.prompt = raw_task.get("prompt", "")
                self.ground_truth = raw_task.get("answer", "")
        else:
            self.prompt = ""
            self.ground_truth = ""

    def _reset_for_rollout(self):
        """Reset internal state for a new rollout."""
        self.done = False
        self.final_reward = 0.0
        self.predicted_answer = ""
        system_prompt = self.dapo_system_template.render()
        formatted_prompt = utils.format_dapo_prompt(self.prompt, attempt=0)
        self.memory = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_prompt},
        ]

    # --- RewardPropagationWorkflow interface ---

    def step(self, step_num: int) -> bool:
        """Execute one problem solving attempt."""
        if self.done:
            return False

        responses = self.model.chat(
            self.memory,
            n=1,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        if responses[0].tokens.shape[0] >= 20480 - 4096:
            self.done = True
            return False

        response_text = responses[0].response_text.strip()
        self.memory.append({"role": "assistant", "content": response_text})

        think, predicted_answer = utils.parse_response(response_text)
        if think is None or predicted_answer is None:
            self.done = True
            return False

        is_correct = utils.my_math_verify(predicted_answer, self.ground_truth)

        if is_correct:
            self.done = True
            self.final_reward = 1.0
            self.predicted_answer = predicted_answer
            feedback = f"Correct! Your answer {predicted_answer} matches the expected answer."
            self.memory.append({"role": "user", "content": f"Feedback: {feedback}"})
            return False
        else:
            self.predicted_answer = predicted_answer
            if step_num < self.max_attempts - 1:
                feedback = (
                    f"Incorrect. Your answer {predicted_answer} does not match. Please try again."
                )
                formatted_feedback = utils.format_dapo_prompt(
                    "", attempt=step_num + 1, feedback=feedback
                )
                self.memory.append({"role": "user", "content": formatted_feedback})
                return True
            else:
                feedback = (
                    f"Incorrect. Your answer {predicted_answer} does not match the expected "
                    f"answer. Maximum attempts reached."
                )
                self.memory.append({"role": "user", "content": f"Feedback: {feedback}"})
                self.done = True
                return False

    def reward(self, exps: list[Experience]) -> float:
        return self.final_reward

    @property
    def max_step_num(self) -> int:
        return self.max_attempts

    # --- Single rollout with step-level decomposition ---

    def _run_single_rollout(self, run_id: int) -> List[Experience]:
        """Run one rollout, returning step-level experiences."""
        self._reset_for_rollout()
        try:
            exps = super().run()
        except Exception as e:
            print(f"[{self.__class__.__name__}] Rollout failed: {e}")
            return []

        for exp in exps:
            exp.eid.run = run_id
            exp.eid.task = str(self.task.task_id)
            if exp.metrics is None:
                exp.metrics = {}
            exp.metrics["success"] = 1.0 if self.final_reward >= 1.0 else 0.0
            exp.metrics["reward"] = self.final_reward

        return exps

    # --- Main entry points ---

    def _eval(self) -> List[Experience]:
        try:
            exps = self._run_single_rollout(run_id=0)
            if not exps:
                return [copy.deepcopy(self.default_exp)]
            return exps
        except Exception:
            return [copy.deepcopy(self.default_exp)]

    def run(self) -> List[Experience]:
        if self.is_eval:
            return self._eval()

        all_exps = []
        for i in range(self.n):
            run_exps = self._run_single_rollout(run_id=i + self.run_id_base)
            if run_exps:
                all_exps.extend(run_exps)
                steps = max(e.eid.step for e in run_exps) + 1
                print(
                    f"[{self.__class__.__name__}] Rollout {i} - "
                    f"reward: {self.final_reward}, attempts: {steps}"
                )
        return all_exps

    def set_repeat_times(self, repeat_times, run_id_base):
        self.repeat_times = repeat_times
        self.run_id_base = run_id_base
        self.n = repeat_times


@WORKFLOWS.register_module("step_grpo_dapo_workflow")
class StepLevelGRPODapoWorkflow(StepLevelDapoBaseWorkflow):
    """Step-level GRPO baseline for DAPO."""

    pass


@WORKFLOWS.register_module("step_opmd_dapo_workflow")
class StepLevelOPMDDapoWorkflow(StepLevelDapoBaseWorkflow):
    """Step-level OPMD baseline for DAPO."""

    pass


@WORKFLOWS.register_module("step_gspo_dapo_workflow")
class StepLevelGSPODapoWorkflow(StepLevelDapoBaseWorkflow):
    """Step-level GSPO baseline for DAPO."""

    pass


@WORKFLOWS.register_module("step_dapo_dapo_workflow")
class StepLevelDAPODapoWorkflow(StepLevelDapoBaseWorkflow):
    """Step-level DAPO baseline for DAPO."""

    pass
