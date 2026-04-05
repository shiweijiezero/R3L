# -*- coding: utf-8 -*-
"""Step-level Reflect-GRPO workflow for DAPO math with Qwen3 compatibility."""
import json
from pathlib import Path
from typing import List, Optional

from jinja2 import Environment, FileSystemLoader

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.envs.R3L.dapo import utils
from trinity.common.workflows.envs.R3L.dapo.step_grpo_workflow import (
    StepLevelDapoBaseWorkflow,
)
from trinity.common.workflows.workflow import WORKFLOWS, Task


@WORKFLOWS.register_module("step_reflect_grpo_dapo_workflow")
class StepLevelReflectGRPODapoWorkflow(StepLevelDapoBaseWorkflow):
    """Step-level Reflect-GRPO for DAPO.

    Half normal rollouts + half reflection+retry from beginning.
    All rollouts produce step-level experiences.
    """

    def __init__(
        self,
        model: ModelWrapper,
        task: Task,
        auxiliary_models: Optional[List] = None,
    ):
        super().__init__(model=model, task=task, auxiliary_models=auxiliary_models)
        self.max_reflect_tokens = 4096
        prompts_dir = Path(__file__).parent / "prompts"
        jinja_env = Environment(
            loader=FileSystemLoader(str(prompts_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.reflection_template = jinja_env.get_template("reflection.j2")

    def _get_reflect(self, trajectory):
        """Generate reflection on a trajectory."""
        formatted_trajectory = utils.format_trajectory_for_reflection(trajectory)
        reflect_prompt = self.reflection_template.render()
        try:
            responses = self.model.chat(
                [
                    {"role": "system", "content": reflect_prompt},
                    {
                        "role": "user",
                        "content": "Here is last attempt trajectory log: \n\n"
                        + formatted_trajectory
                        + "\n\nPlease output in the specified JSON format.",
                    },
                ],
                n=1,
                temperature=self.temperature,
                max_tokens=self.max_reflect_tokens,
            )
            reflection_text = responses[0].response_text.strip()
            first_brace = reflection_text.find("{")
            last_brace = reflection_text.rfind("}")
            if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
                json_str = reflection_text[first_brace : last_brace + 1]
            else:
                json_str = reflection_text
            reflect_resp_tokens = float(len(responses[0].tokens) - responses[0].prompt_length)
            return json.loads(json_str), reflection_text, reflect_resp_tokens
        except Exception as e:
            print(f"[StepReflectGRPO-DAPO] Reflection failed: {e}")
            return None, None, 0.0

    def _run_retry_from_beginning(self, guidance_prompt, run_id):
        """Run a retry rollout from the beginning with guidance."""
        self._reset_for_rollout()
        original_system = self.dapo_system_template.render()
        # Replace system prompt with guided version
        self.memory[0] = {
            "role": "system",
            "content": f"{original_system}\n\n# Previous Attempt Analysis & Guidance\n{guidance_prompt}",
        }
        try:
            exps = super(StepLevelDapoBaseWorkflow, self).run()
        except Exception as e:
            print(f"[StepReflectGRPO-DAPO] Retry rollout failed: {e}")
            return []

        for exp in exps:
            exp.eid.run = run_id
            exp.eid.task = str(self.task.task_id)
            if exp.metrics is None:
                exp.metrics = {}
            exp.metrics["success"] = 1.0 if self.final_reward >= 1.0 else 0.0
            exp.metrics["reward"] = self.final_reward
        return exps

    def run(self) -> List[Experience]:
        if self.is_eval:
            return self._eval()

        all_exps = []
        run_counter = self.run_id_base

        # First half: normal rollouts
        for i in range(self.n // 2):
            run_exps = self._run_single_rollout(run_id=run_counter)
            run_counter += 1
            if run_exps:
                all_exps.extend(run_exps)
                print(
                    f"[StepReflectGRPO-DAPO] Rollout {i} - reward: {self.final_reward}"
                )

        # Second half: first attempt + reflection + retry from beginning
        for i in range(self.n // 2):
            base_exps = self._run_single_rollout(run_id=run_counter)
            run_counter += 1
            base_reward = self.final_reward
            base_trajectory = list(self.memory)

            if base_exps:
                all_exps.extend(base_exps)
                print(
                    f"[StepReflectGRPO-DAPO] First attempt {i} - reward: {base_reward}"
                )

            if base_reward < 1.0:
                reflect_data, reflection_text, reflect_resp_tokens = self._get_reflect(
                    base_trajectory
                )
                # For DAPO, total_steps = number of attempts used
                total_attempts = max(e.eid.step for e in base_exps) + 1 if base_exps else 1
                is_valid, is_perfect = utils.validate_reflect_report(
                    reflect_data, total_attempts
                )
                if is_valid and not is_perfect:
                    guidance = utils.reflect_report_to_guidance_prompt(reflect_data)
                    retry_exps = self._run_retry_from_beginning(guidance, run_id=run_counter)
                    run_counter += 1
                    if retry_exps:
                        for exp in retry_exps:
                            if exp.metrics is None:
                                exp.metrics = {}
                            exp.metrics["reflect_tokens"] = reflect_resp_tokens
                        all_exps.extend(retry_exps)
                        print(
                            f"[StepReflectGRPO-DAPO] Retry {i} - reward: {self.final_reward}, "
                            f"improved: {self.final_reward > base_reward}"
                        )

        return all_exps
