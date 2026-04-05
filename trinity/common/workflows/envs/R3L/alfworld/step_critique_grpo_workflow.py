# -*- coding: utf-8 -*-
"""Step-level Critique-GRPO workflow for ALFWorld with Qwen3 compatibility."""
from pathlib import Path
from typing import Any, List, Optional, Tuple

from jinja2 import Environment, FileSystemLoader

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.envs.R3L.alfworld import utils
from trinity.common.workflows.envs.R3L.alfworld.step_grpo_workflow import (
    StepLevelAlfworldBaseWorkflow,
)
from trinity.common.workflows.workflow import WORKFLOWS, Task


@WORKFLOWS.register_module("step_critique_grpo_alfworld_workflow")
class StepLevelCritiqueGRPOAlfworldWorkflow(StepLevelAlfworldBaseWorkflow):
    """Step-level Critique-GRPO for ALFWorld.

    1. Generate N initial rollouts (step-level)
    2. For each failed rollout, generate critique + refinement (re-execute with guidance)
    3. Select best refinement
    4. Return N runs of step-level experiences: 1 refined (off-policy) + (N-1) initial
    """

    def __init__(
        self,
        model: ModelWrapper,
        task: Task,
        auxiliary_models: Optional[List] = None,
    ):
        super().__init__(model=model, task=task, auxiliary_models=auxiliary_models)
        self.max_critique_tokens = 1024
        prompts_dir = Path(__file__).parent / "prompts"
        jinja_env = Environment(
            loader=FileSystemLoader(str(prompts_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.critique_template = jinja_env.get_template("critique.j2")

    def _generate_critique(self, trajectory, reward) -> Tuple[Optional[str], Optional[Any]]:
        """Generate a critique for a failed trajectory."""
        formatted_trajectory = utils.format_trajectory_for_reflection(trajectory)
        critique_prompt = self.critique_template.render(
            trajectory=formatted_trajectory, reward=reward
        )
        try:
            responses = self.model.chat(
                [{"role": "user", "content": critique_prompt}],
                n=1,
                temperature=0.7,
                max_tokens=self.max_critique_tokens,
            )
            critique_resp_tokens = float(len(responses[0].tokens) - responses[0].prompt_length)
            return responses[0].response_text.strip(), responses[0], critique_resp_tokens
        except Exception as e:
            print(f"[StepCritiqueGRPO] Critique generation failed: {e}")
            return None, None, 0.0

    def _run_refinement_rollout(self, critique_text, run_id):
        """Run a refinement rollout with critique guidance.

        Re-executes the task from the beginning with the critique as guidance.
        """
        self._reset_for_rollout()
        original_system = self.alfworld_system_template.render()
        guidance = (
            f"# Previous Attempt Analysis\n{critique_text}\n\n"
            f"# Instructions\nUse the above analysis to avoid similar mistakes. "
            f"Focus on efficient task completion."
        )
        self.memory = [
            {"role": "system", "content": f"{original_system}\n\n{guidance}"}
        ]
        try:
            exps = super(StepLevelAlfworldBaseWorkflow, self).run()
        except Exception as e:
            print(f"[StepCritiqueGRPO] Refinement rollout failed: {e}")
            return [], 0.0

        for exp in exps:
            exp.eid.run = run_id
            exp.eid.task = str(self.task.task_id)
            exp.info["is_off_policy"] = True
            if exp.metrics is None:
                exp.metrics = {}
            exp.metrics["success"] = 1.0 if self.final_reward >= 1.0 else 0.0
            exp.metrics["reward"] = self.final_reward
        return exps, self.final_reward

    def run(self) -> List[Experience]:
        if self.is_eval:
            return self._eval()

        run_counter = self.run_id_base

        # Step 1: Generate N initial rollouts
        initial_results = []
        for i in range(self.n):
            run_exps = self._run_single_rollout(run_id=run_counter)
            run_counter += 1
            trajectory = list(self.memory)
            reward = self.final_reward

            if run_exps:
                # Mark as on-policy
                for exp in run_exps:
                    exp.eid.task = str(self.task.task_id)
                    exp.info["is_off_policy"] = False
                initial_results.append({
                    "exps": run_exps,
                    "trajectory": trajectory,
                    "reward": reward,
                })
                print(f"[StepCritiqueGRPO] Initial {i+1}/{self.n} - reward: {reward}")

        if not initial_results:
            return []

        # Step 2: For each failed initial, generate critique + refinement
        refinements = []
        for i, result in enumerate(initial_results):
            if result["reward"] >= 1.0:
                continue
            critique_text, _, critique_resp_tokens = self._generate_critique(
                result["trajectory"], result["reward"]
            )
            if critique_text is None:
                continue

            ref_exps, ref_reward = self._run_refinement_rollout(
                critique_text, run_id=run_counter
            )
            run_counter += 1

            if ref_exps and ref_reward > result["reward"]:
                for exp in ref_exps:
                    if exp.metrics is None:
                        exp.metrics = {}
                    exp.metrics["critique_tokens"] = critique_resp_tokens
                refinements.append({
                    "exps": ref_exps,
                    "reward": ref_reward,
                    "original_idx": i,
                })
                print(
                    f"[StepCritiqueGRPO] Refinement for initial {i+1} - "
                    f"reward: {ref_reward}"
                )

        # Step 3: Select best refinement
        best_refinement = None
        if refinements:
            refinements.sort(key=lambda x: x["reward"], reverse=True)
            for ref in refinements:
                if ref["reward"] >= 1.0:
                    best_refinement = ref
                    break
            if best_refinement is None:
                best_refinement = refinements[0]

        # Step 4: Construct final experience list
        exp_lst = []
        if best_refinement is not None:
            exp_lst.extend(best_refinement["exps"])
            used_runs = 1
            for i, result in enumerate(initial_results):
                if i == best_refinement["original_idx"]:
                    continue
                if used_runs >= self.n:
                    break
                exp_lst.extend(result["exps"])
                used_runs += 1
        else:
            for result in initial_results[:self.n]:
                exp_lst.extend(result["exps"])

        off_policy_count = sum(
            1 for exp in exp_lst if exp.info.get("is_off_policy", False)
        )
        print(
            f"[StepCritiqueGRPO Summary] {len(exp_lst)} step-experiences, "
            f"{off_policy_count} off-policy"
        )
        return exp_lst
