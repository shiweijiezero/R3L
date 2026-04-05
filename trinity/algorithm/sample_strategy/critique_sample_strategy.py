"""Sample strategy for Critique-GRPO that passes is_off_policy custom field."""

from typing import Dict, List, Tuple

import torch

from trinity.algorithm.sample_strategy.sample_strategy import (
    SAMPLE_STRATEGY,
    SampleStrategy,
)
from trinity.algorithm.sample_strategy.utils import representative_sample
from trinity.buffer import get_buffer_reader
from trinity.common.config import BufferConfig
from trinity.common.experience import CustomField, Experiences
from trinity.utils.timer import Timer


@SAMPLE_STRATEGY.register_module("critique_grpo")
class CritiqueGRPOSampleStrategy(SampleStrategy):
    """Sample strategy for Critique-GRPO.

    This strategy extracts the is_off_policy field from experience.info
    and passes it to the policy loss function for shaping computation.
    """

    def __init__(self, buffer_config: BufferConfig, **kwargs):
        super().__init__(buffer_config)
        self.exp_buffer = get_buffer_reader(buffer_config.trainer_input.experience_buffer)

    async def sample(self, step: int, **kwargs) -> Tuple[Experiences, Dict, List]:
        metrics = {}
        with Timer(metrics, "time/read_experience"):
            exp_list = await self.exp_buffer.read_async()
            repr_samples = representative_sample(exp_list)

        # Count off-policy samples for logging
        off_policy_count = sum(1 for exp in exp_list if exp.info.get("is_off_policy", False))
        metrics["critique/off_policy_ratio"] = off_policy_count / len(exp_list) if exp_list else 0.0

        self.set_model_version_metric(exp_list, metrics)
        with Timer(metrics, "time/gather_experience"):
            exps = Experiences.gather_experiences(
                experiences=exp_list,
                pad_token_id=self.pad_token_id,
                custom_fields=[
                    CustomField(
                        source_field="is_off_policy",
                        destination_field="is_off_policy",
                        data_type=torch.bool,
                    ),
                ],
            )
        return exps, metrics, repr_samples

    @classmethod
    def default_args(cls) -> dict:
        return {}
