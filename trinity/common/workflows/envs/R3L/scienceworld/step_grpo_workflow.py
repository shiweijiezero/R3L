# -*- coding: utf-8 -*-
"""Step-level GRPO/OPMD/GSPO workflows for ScienceWorld with Qwen3 compatibility.

Qwen3's chat template removes thinking tokens from context between turns (dynamic context),
so multi-turn conversations cannot be concatenated as a single training sequence.
These workflows decompose each turn into an independent Experience via
RewardPropagationWorkflow, enabling step-level training.
"""
import copy
from pathlib import Path
from typing import List, Optional

import torch
from jinja2 import Environment, FileSystemLoader

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.envs.R3L.scienceworld import utils
from trinity.common.workflows.step_wise_workflow import RewardPropagationWorkflow
from trinity.common.workflows.workflow import WORKFLOWS, Task


class StepLevelScienceWorldBaseWorkflow(RewardPropagationWorkflow):
    """Base step-level workflow for ScienceWorld.

    Each environment interaction step produces a separate Experience via
    model.extract_experience_from_history(). All steps in a rollout share
    the same final reward (reward propagation).

    ScienceWorld rewards are 0-100 and normalized to 0-1. The maximum reward
    seen during a rollout is used as the final reward.
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
        self.max_env_steps = 30
        self.max_tokens = 16384
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
        self.sciworld_system_template = self.jinja_env.get_template("sciworld_system.j2")

        # Default experience for error/empty cases
        self.default_exp = Experience(
            tokens=torch.tensor([0, 0], dtype=torch.long),
            prompt_length=1,
            action_mask=torch.tensor([False], dtype=torch.bool),
            logprobs=torch.tensor([0.0], dtype=torch.float),
            metrics={"success": 0.0, "reward": 0.0},
            reward=0.0,
        )

        # Per-rollout state (reset before each rollout)
        self.env = None
        self.observation = None
        self.action_history = []
        self.done = False
        self.final_reward = 0.0
        self.current_reward = 0.0
        self.memory = []

        print(f"Initializing {self.__class__.__name__}, temperature={self.temperature}")
        self.reset(task)

    def reset(self, task: Task):
        self.task_desc = task.task_desc or "0"
        self.is_eval = task.is_eval
        self.temperature = getattr(task.rollout_args, "temperature", 1.0)
        self.task = task
        self.n = task.repeat_times

    def _reset_for_rollout(self):
        """Reset internal state for a new rollout."""
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
        self.env = utils.create_sciworld_environment(self.task_desc)
        observation, info = self.env.reset()
        self.observation = (
            "Task Description: " + str(self.env.get_task_description()) + "\n" + observation
        )
        self.action_history = []
        self.done = False
        self.final_reward = 0.0
        self.current_reward = 0.0
        self.memory = [{"role": "system", "content": self.sciworld_system_template.render()}]

    # --- RewardPropagationWorkflow interface ---

    def step(self, step_num: int) -> bool:
        """Execute one environment interaction step."""
        if self.done:
            return False

        formatted_obs = utils.format_observation(self.observation)
        self.memory.append({"role": "user", "content": formatted_obs})

        responses = self.model.chat(
            self.memory,
            n=1,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        if responses[0].tokens.shape[0] >= 20480 - 512:
            self.done = True
            return False

        response_text = responses[0].response_text.strip()
        self.memory.append({"role": "assistant", "content": response_text})

        think, action = utils.parse_response(response_text)
        if action is None:
            self.done = True
            return False

        # Consecutive repetition check (last 3 actions)
        self.action_history.append(action)
        if len(self.action_history) > 3:
            self.action_history.pop(0)
        if len(self.action_history) >= 3 and all(
            a == self.action_history[-1] for a in self.action_history[-3:]
        ):
            self.done = True
            return False

        action_valid, error_msg = utils.validate_action(action)
        if not action_valid:
            self.done = True
            return False

        observation, reward, done, info = self.env.step(action)
        # Track maximum reward (ScienceWorld gives 0-100)
        if reward > self.current_reward:
            self.final_reward = reward
            self.current_reward = reward

        self.observation = observation
        if done:
            self.done = True
            return False

        return True

    def reward(self, exps: list[Experience]) -> float:
        # Normalize from 0-100 to 0-1
        return self.final_reward / 100.0

    @property
    def max_step_num(self) -> int:
        return self.max_env_steps

    # --- Single rollout with step-level decomposition ---

    def _run_single_rollout(self, run_id: int) -> List[Experience]:
        """Run one rollout, returning step-level experiences."""
        self._reset_for_rollout()
        try:
            exps = super().run()
        except Exception as e:
            print(f"[{self.__class__.__name__}] Rollout failed: {e}")
            return []

        normalized_reward = self.final_reward / 100.0
        for exp in exps:
            exp.eid.run = run_id
            exp.eid.task = str(self.task.task_id)
            if exp.metrics is None:
                exp.metrics = {}
            exp.metrics["success"] = 1.0 if normalized_reward >= 1.0 else 0.0
            exp.metrics["reward"] = normalized_reward

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
                normalized = self.final_reward / 100.0
                print(
                    f"[{self.__class__.__name__}] Rollout {i} - "
                    f"reward: {normalized}, steps: {steps}"
                )
        return all_exps

    def set_repeat_times(self, repeat_times, run_id_base):
        self.repeat_times = repeat_times
        self.run_id_base = run_id_base
        self.n = repeat_times

    def __del__(self):
        if hasattr(self, "env") and self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass


@WORKFLOWS.register_module("step_grpo_scienceworld_workflow")
class StepLevelGRPOScienceWorldWorkflow(StepLevelScienceWorldBaseWorkflow):
    """Step-level GRPO baseline for ScienceWorld."""

    pass


@WORKFLOWS.register_module("step_opmd_scienceworld_workflow")
class StepLevelOPMDScienceWorldWorkflow(StepLevelScienceWorldBaseWorkflow):
    """Step-level OPMD baseline for ScienceWorld."""

    pass


@WORKFLOWS.register_module("step_gspo_scienceworld_workflow")
class StepLevelGSPOScienceWorldWorkflow(StepLevelScienceWorldBaseWorkflow):
    """Step-level GSPO baseline for ScienceWorld."""

    pass


@WORKFLOWS.register_module("step_dapo_scienceworld_workflow")
class StepLevelDAPOScienceWorldWorkflow(StepLevelScienceWorldBaseWorkflow):
    """Step-level DAPO baseline for ScienceWorld."""

    pass
