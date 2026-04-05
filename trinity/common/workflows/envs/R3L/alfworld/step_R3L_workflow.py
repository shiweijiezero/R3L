# -*- coding: utf-8 -*-
"""Step-level R3L (Reflect-then-Retry RL) workflow for ALFWorld with Qwen3 compatibility.

Key design for step-level R3L:
- N/2 base rollouts + N/2 retry rollouts
- Base and retry share the same eid.task ("_explore") for group-level advantage comparison
- Retry starts from pivot point: only steps >= pivot produce training experiences
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
from trinity.common.workflows.envs.R3L.alfworld import utils
from trinity.common.workflows.envs.R3L.alfworld.step_grpo_workflow import (
    StepLevelAlfworldBaseWorkflow,
)
from trinity.common.workflows.workflow import WORKFLOWS, Task


@WORKFLOWS.register_module("step_R3L_alfworld_workflow")
class StepLevelR3LAlfworldWorkflow(StepLevelAlfworldBaseWorkflow):
    """Step-level R3L for ALFWorld.

    Each rollout is decomposed into step-level experiences. Base and retry
    rollouts share the same task group for advantage comparison. The retry
    only produces experiences from the pivot point onwards.
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
            print(f"[StepR3L] Reflection failed: {e}")
            return None, None, None, 0.0

    def _run_retry_from_pivot(
        self, guidance_prompt, base_trajectory, pivot_step, run_id
    ) -> Tuple[List[Experience], float]:
        """Run a retry rollout from the pivot point with guidance.

        Steps 0..pivot-1: Replay environment using base trajectory actions (no model calls).
        Steps pivot..end: Generate new model responses with guidance (produce experiences).

        Args:
            guidance_prompt: Guidance text from reflection.
            base_trajectory: The base attempt's conversation history.
            pivot_step: The step to start retry from.
            run_id: Run ID for the retry experiences.

        Returns:
            Tuple of (step-level experiences from pivot onwards, final reward).
        """
        # Reset environment
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
        self.env = utils.create_alfworld_environment(self.game_file_path)
        observation, info = self.env.reset()
        task_description = utils.extract_task_description(observation)

        original_system = self.alfworld_system_template.render()
        action_history = []

        # Build conversation history by replaying base trajectory up to pivot
        if pivot_step > 0:
            memory = [{"role": "system", "content": original_system}]
            first_step = 0
            for msg in base_trajectory[1:]:  # Skip system message
                if msg["role"] == "user":
                    memory.append(msg)
                elif msg["role"] == "assistant":
                    if first_step < pivot_step:
                        memory.append(msg)
                        # Replay action in environment
                        think, action, _ = utils.parse_response(msg["content"])
                        if think is not None and action is not None:
                            observation, reward, done, info = self.env.step(action)
                            action_history.append(action)
                        first_step += 1
                        if done:
                            return [], reward
                    else:
                        break

            # Add guidance as a system message at the pivot point
            guidance_msg = {
                "role": "system",
                "content": f"# Previous Attempt Analysis & Guidance\n{guidance_prompt}",
            }
            memory.append(guidance_msg)
        else:
            # Retry from beginning with guidance in system prompt
            merged_system = (
                f"{original_system}\n\n"
                f"# Previous Attempt Analysis & Guidance\n{guidance_prompt}"
            )
            memory = [{"role": "system", "content": merged_system}]

        # Now generate new responses from pivot onwards using model.chat
        self.observation = observation
        self.info = info
        self.task_description = task_description
        self.action_history = action_history
        self.done = False
        self.final_reward = 0.0
        self.memory = memory

        # Run step-level generation from pivot to max_env_steps
        experiences = []
        start_step = pivot_step if pivot_step > 0 else 0
        last_step = start_step
        for step_num in range(start_step, self.max_env_steps):
            if self.done:
                break

            last_step = step_num
            continue_run = self.step(step_num=step_num)

            # Extract step-level experience
            exps = self.model.extract_experience_from_history()
            for exp in exps:
                exp.eid.step = step_num
                exp.eid.run = run_id
            experiences.extend(exps)

            if not continue_run:
                break

        # Assign final reward and token metrics to all retry step experiences
        final_reward = self.final_reward
        for exp in experiences:
            exp.reward = final_reward
            if exp.metrics is None:
                exp.metrics = {}
            exp.metrics["actual_env_steps"] = last_step + 1
            exp.metrics["response_tokens"] = float(len(exp.tokens) - exp.prompt_length)
            exp.metrics["prompt_tokens"] = float(exp.prompt_length)
        if experiences:
            traj_resp_tokens = sum(len(e.tokens) - e.prompt_length for e in experiences)
            experiences[-1].metrics["trajectory_response_tokens"] = float(traj_resp_tokens)

        return experiences, final_reward

    def _run_retry_distill_from_pivot(
        self, guidance_prompt, base_trajectory, pivot_step, run_id
    ) -> Tuple[List[Experience], float]:
        """Run a retry rollout from pivot for the distill trajectory.

        The model generates with guidance in context (to produce good retry actions),
        but in step-level training each step's experience only contains that turn's
        tokens. The guidance is part of the prompt prefix and won't be trained on.

        This is the trajectory that enters RL training (the _explore group).
        """
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
        self.env = utils.create_alfworld_environment(self.game_file_path)
        observation, info = self.env.reset()
        task_description = utils.extract_task_description(observation)

        original_system = self.alfworld_system_template.render()
        action_history = []

        # We run the model with guidance but produce two trajectories:
        # 1. The actual trajectory (with guidance) - for environment interaction
        # 2. The distill trajectory (without guidance) - for experience extraction
        # Since step-level extraction uses model.extract_experience_from_history(),
        # and the model sees the guided context, we use a two-model approach:
        # - Use the guided context for model.chat() (to get good retry actions)
        # - But the experience is extracted from this chat (which includes guidance in context)
        #
        # For step-level training, this is acceptable because each step's experience
        # only contains the current turn's tokens, not the full context.
        # The guidance in the context is part of the "prompt" and won't be trained on.

        # Build replay memory
        guided_memory = [{"role": "system", "content": original_system}]

        if pivot_step > 0:
            first_step = 0
            for msg in base_trajectory[1:]:
                if msg["role"] == "user":
                    guided_memory.append(msg)
                elif msg["role"] == "assistant":
                    if first_step < pivot_step:
                        guided_memory.append(msg)
                        think, action, _ = utils.parse_response(msg["content"])
                        if think is not None and action is not None:
                            observation, reward, done, info = self.env.step(action)
                            action_history.append(action)
                        first_step += 1
                        if done:
                            return [], reward
                    else:
                        break
            # Add guidance
            guided_memory.append({
                "role": "system",
                "content": f"# Previous Attempt Analysis & Guidance\n{guidance_prompt}",
            })
        else:
            merged_system = (
                f"{original_system}\n\n"
                f"# Previous Attempt Analysis & Guidance\n{guidance_prompt}"
            )
            guided_memory = [{"role": "system", "content": merged_system}]

        self.observation = observation
        self.info = info
        self.task_description = task_description
        self.action_history = action_history
        self.done = False
        self.final_reward = 0.0
        self.memory = guided_memory

        experiences = []
        start_step = pivot_step if pivot_step > 0 else 0
        last_step = start_step
        for step_num in range(start_step, self.max_env_steps):
            if self.done:
                break

            last_step = step_num
            continue_run = self.step(step_num=step_num)
            exps = self.model.extract_experience_from_history()
            for exp in exps:
                exp.eid.step = step_num
                exp.eid.run = run_id
            experiences.extend(exps)

            if not continue_run:
                break

        final_reward = self.final_reward
        for exp in experiences:
            exp.reward = final_reward
            if exp.metrics is None:
                exp.metrics = {}
            exp.metrics["actual_env_steps"] = last_step + 1
            exp.metrics["response_tokens"] = float(len(exp.tokens) - exp.prompt_length)
            exp.metrics["prompt_tokens"] = float(exp.prompt_length)
        if experiences:
            traj_resp_tokens = sum(len(e.tokens) - e.prompt_length for e in experiences)
            experiences[-1].metrics["trajectory_response_tokens"] = float(traj_resp_tokens)

        return experiences, final_reward

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
                base_steps = max(e.eid.step for e in base_exps) + 1 if base_exps else 0

                # Set task group for explore (base + retry share this group)
                for exp in base_exps:
                    exp.eid.task = task_id + "_explore"
                    exp.metrics["is_base_rollout"] = 1.0
                    exp.metrics["is_retry_rollout"] = 0.0
                all_exps.extend(base_exps)
                print(f"[StepR3L] Base rollout {i} - reward: {base_reward}, steps: {base_steps}")

                # === Reflection ===
                reflect_data, reflection_text, reflect_exp, reflect_resp_tokens = self._get_reflect(
                    base_trajectory
                )
                is_valid, is_perfect = utils.validate_reflect_report(reflect_data, base_steps)

                if not is_valid or is_perfect:
                    # Perfect trajectory or invalid reflection
                    if base_reward >= 1.0 and is_perfect and reflect_exp is not None:
                        reflect_exp.reward = 1.0
                        reflect_exp.eid.task = task_id + f"_reflect_{i}"
                        reflect_exp.eid.run = run_counter
                        run_counter += 1
                        reflect_exp.metrics = reflect_exp.metrics if reflect_exp.metrics else {}
                        reflect_exp.metrics["reflect_tokens"] = reflect_resp_tokens
                        all_exps.append(reflect_exp)
                    continue

                guidance_prompt = utils.reflect_report_to_guidance_prompt(reflect_data)
                pivot_step = reflect_data.get("retry_from_step", 0)
                print(f"[StepR3L] Pivot point: {pivot_step}")

                # === Retry rollout from pivot ===
                retry_run_id = run_counter
                run_counter += 1

                retry_exps, retry_reward = self._run_retry_distill_from_pivot(
                    guidance_prompt, base_trajectory, pivot_step, retry_run_id
                )

                # Set task group for explore (same group as base)
                for exp in retry_exps:
                    exp.eid.task = task_id + "_explore"
                    if exp.metrics is None:
                        exp.metrics = {}
                    exp.metrics["second_reward"] = retry_reward
                    exp.metrics["second_improve"] = 1.0 if retry_reward > base_reward else 0.0
                    exp.metrics["pivot_point"] = pivot_step
                    exp.metrics["is_retry_rollout"] = 1.0
                    exp.metrics["is_base_rollout"] = 0.0
                    exp.metrics["reflect_tokens"] = reflect_resp_tokens

                all_exps.extend(retry_exps)
                print(
                    f"[StepR3L] Retry rollout {i} - reward: {retry_reward}, "
                    f"improve: {retry_reward > base_reward}"
                )

                # === Credit masking for base rollout ===
                # In step-level mode, credit masking means we zero out the
                # action_mask for base steps >= pivot, so the trainer will not
                # compute loss on those divergent steps (only the retry version
                # is trained on). Steps < pivot are shared prefix and are kept.
                if pivot_step > 0:
                    for exp in base_exps:
                        if exp.eid.step >= pivot_step:
                            exp.action_mask = exp.action_mask * 0

                # === SFT data: if retry improved, record reflect + retry for SFT ===
                if (retry_reward > base_reward and retry_reward >= 1.0) or (
                    retry_reward >= 1.0 and base_reward < 1.0
                ):
                    # Record reflect exp for SFT
                    if reflect_exp is not None:
                        reflect_exp.reward = 1.0
                        reflect_exp.eid.task = task_id + f"_reflect_{i}"
                        reflect_exp.eid.run = run_counter
                        run_counter += 1
                        reflect_exp.metrics = reflect_exp.metrics if reflect_exp.metrics else {}
                        reflect_exp.metrics["reflect_tokens"] = reflect_resp_tokens
                        all_exps.append(reflect_exp)

                    # Record retry with guidance as SFT target (separate group)
                    sft_retry_exps, _ = self._run_retry_from_pivot(
                        guidance_prompt, base_trajectory, pivot_step,
                        run_id=run_counter
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
                        print("[StepR3L] Recorded reflect + retry SFT data")

            except Exception as e:
                print(f"[StepR3L] Iteration {i} failed: {e}")

        return all_exps
