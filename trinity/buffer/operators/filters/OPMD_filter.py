from typing import List, Tuple

import copy

from trinity.buffer.operators import EXPERIENCE_OPERATORS, ExperienceOperator
from trinity.common.experience import Experience, group_by


@EXPERIENCE_OPERATORS.register_module("OPMD_filter")
class OPMDFilter(ExperienceOperator):
    def __init__(self):
        pass

    def process(self, exps: List[Experience]) -> Tuple[List[Experience], dict]:
        # 过滤无效的经验
        filtered_exps = [exp for exp in exps if exp.reward is not None and exp.prompt_length > 2]
        metrics = {"filtered_count": len(exps) - len(filtered_exps)}
        return filtered_exps, metrics

