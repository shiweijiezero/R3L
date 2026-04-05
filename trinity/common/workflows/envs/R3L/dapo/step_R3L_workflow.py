# -*- coding: utf-8 -*-
"""Step-level R3L (Reflect-then-Retry RL) workflow for DAPO math with Qwen3 compatibility.

Key design for step-level R3L:
- N/2 base rollouts + N/2 retry rollouts
- Base and retry share the same eid.task ("_explore") for group-level advantage comparison
- For DAPO math, pivot always starts from attempt 0 (retry from beginning)
- Reflect experiences are in separate eid.task groups ("_reflect_i")
- SFT experiences (successful reflect+retry) are in separate groups ("_retry_i")

EID structure (run IDs are auto-incremented via run_counter):
    Base rollout i, step j:  eid.task = task_id + "_explore", eid.run = run_counter++, eid.step = j
    Retry rollout i, step j: eid.task = task_id + "_explore", eid.run = run_counter++, eid.step = j
    Reflect exp i:           eid.task = task_id + "_reflect_{i}", eid.run = run_counter++
    SFT retry exp i:         eid.task = task_id + "_retry_{i}",  eid.run = run_counter++
"""
import json
from pathlib import Path
from typing import List, Optional, Tuple

from jinja2 import Environment, FileSystemLoader

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.envs.R3L.dapo import utils
from trinity.common.workflows.envs.R3L.dapo.step_grpo_workflow import (
    StepLevelDapoBaseWorkflow,
)
from trinity.common.workflows.workflow import WORKFLOWS, Task


@WORKFLOWS.register_module("step_R3L_dapo_workflow")
class StepLevelR3LDapoWorkflow(StepLevelDapoBaseWorkflow):
    """Step-level R3L for DAPO.

    Each rollout is decomposed into step-level experiences. Base and retry
    rollouts share the same task group for advantage comparison.
    For math problems, retry always starts from the beginning with guidance.
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
        """Generate reflection report from a trajectory."""
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
            reflection_data = json.loads(json_str)
            reflect_resp_tokens = float(len(responses[0].tokens) - responses[0].prompt_length)
            return reflection_data, reflection_text, responses[0], reflect_resp_tokens
        except Exception as e:
            print(f"[StepR3L-DAPO] Reflection failed: {e}")
            return None, None, None, 0.0

    def _run_retry_from_beginning(self, guidance_prompt, run_id):
        """Run retry with guidance (used for SFT data)."""
        self._reset_for_rollout()
        original_system = self.dapo_system_template.render()
        self.memory[0] = {
            "role": "system",
            "content": f"{original_system}\n\n# Previous Attempt Analysis & Guidance\n{guidance_prompt}",
        }
        try:
            exps = super(StepLevelDapoBaseWorkflow, self).run()
        except Exception as e:
            print(f"[StepR3L-DAPO] Retry rollout failed: {e}")
            return [], 0.0

        for exp in exps:
            exp.eid.run = run_id
            if exp.metrics is None:
                exp.metrics = {}
            exp.metrics["success"] = 1.0 if self.final_reward >= 1.0 else 0.0
            exp.metrics["reward"] = self.final_reward

        return exps, self.final_reward

    def _run_retry_distill_from_beginning(self, guidance_prompt, run_id):
        """Run retry with guidance for the distill trajectory (enters _explore group).

        The model generates with guidance in context, but in step-level training
        each step's experience only contains that turn's tokens. The guidance is
        part of the prompt prefix and won't be trained on.
        """
        self._reset_for_rollout()
        original_system = self.dapo_system_template.render()
        self.memory[0] = {
            "role": "system",
            "content": f"{original_system}\n\n# Previous Attempt Analysis & Guidance\n{guidance_prompt}",
        }
        try:
            exps = super(StepLevelDapoBaseWorkflow, self).run()
        except Exception as e:
            print(f"[StepR3L-DAPO] Retry distill rollout failed: {e}")
            return [], 0.0

        for exp in exps:
            exp.eid.run = run_id
            if exp.metrics is None:
                exp.metrics = {}
            exp.metrics["success"] = 1.0 if self.final_reward >= 1.0 else 0.0
            exp.metrics["reward"] = self.final_reward

        return exps, self.final_reward

    def run(self) -> List[Experience]:
        if self.is_eval:
            return self._eval()

        all_exps = []
        run_counter = self.run_id_base
        task_id = str(self.task.task_id)

        for i in range(self.n // 2):
            try:
                # === Base rollout ===
                base_run_id = run_counter
                run_counter += 1
                base_exps = self._run_single_rollout(run_id=base_run_id)
                base_reward = self.final_reward
                base_trajectory = list(self.memory)
                base_attempts = max(e.eid.step for e in base_exps) + 1 if base_exps else 1

                for exp in base_exps:
                    exp.eid.task = task_id + "_explore"
                    exp.metrics["is_base_rollout"] = 1.0
                    exp.metrics["is_retry_rollout"] = 0.0

                all_exps.extend(base_exps)
                print(
                    f"[StepR3L-DAPO] Base rollout {i} - "
                    f"reward: {base_reward}, attempts: {base_attempts}"
                )

                # === Reflection ===
                reflect_data, reflection_text, reflect_exp, reflect_resp_tokens = self._get_reflect(
                    base_trajectory
                )
                is_valid, is_perfect = utils.validate_reflect_report(
                    reflect_data, base_attempts
                )

                if not is_valid or is_perfect:
                    if base_reward >= 1.0 and is_perfect and reflect_exp is not None:
                        reflect_exp.reward = 1.0
                        reflect_exp.eid.task = task_id + f"_reflect_{i}"
                        reflect_exp.eid.run = run_counter
                        run_counter += 1
                        if reflect_exp.metrics is None:
                            reflect_exp.metrics = {}
                        reflect_exp.metrics["reflect_tokens"] = reflect_resp_tokens
                        all_exps.append(reflect_exp)
                    continue

                guidance_prompt = utils.reflect_report_to_guidance_prompt(reflect_data)

                # === Retry from beginning (DAPO always retries from scratch) ===
                retry_run_id = run_counter
                run_counter += 1

                retry_exps, retry_reward = self._run_retry_distill_from_beginning(
                    guidance_prompt, retry_run_id
                )

                for exp in retry_exps:
                    exp.eid.task = task_id + "_explore"
                    if exp.metrics is None:
                        exp.metrics = {}
                    exp.metrics["second_reward"] = retry_reward
                    exp.metrics["second_improve"] = (
                        1.0 if retry_reward > base_reward else 0.0
                    )
                    exp.metrics["is_retry_rollout"] = 1.0
                    exp.metrics["is_base_rollout"] = 0.0
                    exp.metrics["reflect_tokens"] = reflect_resp_tokens

                all_exps.extend(retry_exps)
                print(
                    f"[StepR3L-DAPO] Retry rollout {i} - reward: {retry_reward}, "
                    f"improve: {retry_reward > base_reward}"
                )

                # === SFT data ===
                if (retry_reward > base_reward and retry_reward >= 1.0) or (
                    retry_reward >= 1.0 and base_reward < 1.0
                ):
                    if reflect_exp is not None:
                        reflect_exp.reward = 1.0
                        reflect_exp.eid.task = task_id + f"_reflect_{i}"
                        reflect_exp.eid.run = run_counter
                        run_counter += 1
                        if reflect_exp.metrics is None:
                            reflect_exp.metrics = {}
                        reflect_exp.metrics["reflect_tokens"] = reflect_resp_tokens
                        all_exps.append(reflect_exp)

                    sft_retry_exps, _ = self._run_retry_from_beginning(
                        guidance_prompt, run_id=run_counter
                    )
                    run_counter += 1
                    for exp in sft_retry_exps:
                        exp.eid.task = task_id + f"_retry_{i}"
                        exp.reward = 1.0
                        if exp.metrics is None:
                            exp.metrics = {}
                        exp.metrics["is_sft_retry"] = 1.0
                    if sft_retry_exps:
                        all_exps.extend(sft_retry_exps)
                        print("[StepR3L-DAPO] Recorded reflect + retry SFT data")

            except Exception as e:
                print(f"[StepR3L-DAPO] Iteration {i} failed: {e}")

        return all_exps
