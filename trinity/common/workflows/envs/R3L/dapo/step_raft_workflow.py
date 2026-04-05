# -*- coding: utf-8 -*-
"""Step-level RAFT workflow for DAPO math with Qwen3 compatibility."""
import copy
from typing import List, Optional

from trinity.common.experience import Experience
from trinity.common.workflows.envs.R3L.dapo.step_grpo_workflow import (
    StepLevelDapoBaseWorkflow,
)
from trinity.common.workflows.workflow import WORKFLOWS


@WORKFLOWS.register_module("step_raft_dapo_workflow")
class StepLevelRAFTDapoWorkflow(StepLevelDapoBaseWorkflow):
    """Step-level RAFT baseline for DAPO.

    Only keeps step-level experiences from successful rollouts (reward >= 1.0).
    Failed rollouts are replaced with default (empty) experiences.
    """

    def run(self) -> List[Experience]:
        if self.is_eval:
            return self._eval()

        all_exps = []
        for i in range(self.n):
            run_exps = self._run_single_rollout(run_id=i + self.run_id_base)
            if run_exps and self.final_reward >= 1.0:
                all_exps.extend(run_exps)
                steps = max(e.eid.step for e in run_exps) + 1
                print(
                    f"[StepRAFT-DAPO] Rollout {i} - reward: {self.final_reward}, "
                    f"attempts: {steps} (kept)"
                )
            else:
                all_exps.append(copy.deepcopy(self.default_exp))
                print(
                    f"[StepRAFT-DAPO] Rollout {i} - reward: {self.final_reward} (discarded)"
                )
        return all_exps
