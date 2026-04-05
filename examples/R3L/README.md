# R3L Experiments

This directory contains configurations and data for running R3L experiments across multiple environments.

## Directory Structure

```
R3L/
├── alfworld/           # ALFWorld household tasks
│   ├── alfworld_data/  # Task data (train/test splits)
│   └── *.yaml          # Training configs
├── scienceworld/       # ScienceWorld experiments
│   ├── scienceworld_data/
│   └── *.yaml
├── webshop/            # WebShop navigation
│   ├── webshop_data/
│   └── *.yaml
├── dapo/               # DAPO math problems
│   ├── dapo_data/
│   └── *.yaml
└── countdown/          # Countdown game
    ├── countdown_data/
    └── *.yaml
```

## Configuration Naming Convention

| Pattern | Description |
|---------|-------------|
| `opmd_R3L_*.yaml` | **R3L (full)** - Complete algorithm with all components |
| `opmd_R3L_w_o_credit_*.yaml` | R3L **without** Pivotal Credit assignment |
| `opmd_R3L_w_o_reweight_*.yaml` | R3L **without** Positive Amplification |
| `grpo_*.yaml` | GRPO baseline |
| `opmd_*.yaml` | OPMD baseline |
| `RAFT_*.yaml` | RAFT baseline |
| `reflect_grpo_*.yaml` | GRPO + reflection (no retry) |
| `critique_grpo_*.yaml` | GRPO + critique feedback |
| `gspo_*.yaml` | GSPO baseline |
| `dapo_*.yaml` | DAPO baseline |

Model sizes: `1.5B`, `3B`, `7B` (corresponding to Qwen2.5 variants)

## Quick Start

### 1. Environment Setup

```bash
# Activate environment
conda activate r3l

# Set model path
export TRINITY_MODEL_PATH=/path/to/Qwen2.5-1.5B-Instruct

# Set checkpoint directory
export TRINITY_CHECKPOINT_ROOT_DIR=./checkpoints
```

### 2. Start Ray Cluster

```bash
# Single node
ray start --head

# Multi-node (on head node)
ray start --head --port=6379

# Multi-node (on worker nodes)
ray start --address=<head_ip>:6379
```

### 3. Run Training

```bash
# R3L on ALFWorld
trinity run --config examples/R3L/alfworld/opmd_R3L_1.5B.yaml

# R3L on ScienceWorld
trinity run --config examples/R3L/scienceworld/opmd_R3L_1.5B.yaml

# R3L on WebShop
trinity run --config examples/R3L/webshop/opmd_R3L_1.5B.yaml

# R3L on DAPO (Math)
trinity run --config examples/R3L/dapo/opmd_R3L_1.5B.yaml
```

## Environment Details

### ALFWorld

Text-based interactive household environment with 6 task types:
- Pick & Place
- Examine in Light
- Clean & Place
- Heat & Place
- Cool & Place
- Pick Two & Place

**Key parameters:**
- `max_env_steps`: 25 (maximum interaction steps)
- `max_tokens`: 512 (per response)
- `temperature`: 1.0 (training), 0.4 (evaluation)

### ScienceWorld

Scientific reasoning environment requiring multi-step experiments.

**Task categories:**
- Boiling/Melting/Freezing points
- Electrical conductivity
- Friction experiments
- Life cycles
- And more...

### WebShop

Goal-oriented web shopping navigation environment.

**Task:** Find and purchase products matching given criteria (attributes, price constraints).

### DAPO

Mathematical problem-solving benchmark with verifiable answers.

### Countdown

Number game: combine given numbers using arithmetic operations to reach a target.

## Configuration Parameters

### Key Settings

```yaml
algorithm:
  algorithm_type: opmd_reweight_adv  # R3L algorithm variant
  repeat_times: 8                     # Rollouts per task

model:
  max_response_tokens: 512
  max_model_len: 20480

buffer:
  total_epochs: 20
  batch_size: 96
  explorer_input:
    default_workflow_type: 'R3L_alfworld_workflow'
    taskset:
      rollout_args:
        temperature: 1.0

explorer:
  runner_per_model: 32
  eval_interval: 20
  rollout_model:
    engine_num: 5
    gpu_memory_utilization: 0.7
```

### Resource Requirements

| Model Size | Minimum GPUs | Recommended |
|------------|--------------|-------------|
| 1.5B | 4 | 8 |
| 3B | 8 | 8 |
| 7B | 8 | 16 |

## Monitoring

### Wandb

```bash
export WANDB_API_KEY=<your_key>
wandb login
```

Metrics tracked:
- `success`: Task success rate
- `second_success`: Success rate after retry
- `second_improve`: Improvement rate from retry
- `pivot_point`: Average retry step identified
- `reward`, `second_reward`: Reward values

### TensorBoard

```bash
tensorboard --logdir checkpoints/<project>/<name>/logs
```

## Ablation Studies

To run ablation experiments:

```bash
# Full R3L
trinity run --config examples/R3L/alfworld/opmd_R3L_1.5B.yaml

# Without Pivotal Credit
trinity run --config examples/R3L/alfworld/opmd_R3L_w_o_credit_1.5B.yaml

# Without Positive Amplification
trinity run --config examples/R3L/alfworld/opmd_R3L_w_o_reweight_1.5B.yaml

# Baselines
trinity run --config examples/R3L/alfworld/grpo_1.5B.yaml
trinity run --config examples/R3L/alfworld/opmd_1.5B.yaml
```

## Workflow Implementation

R3L workflows are located in `trinity/common/workflows/envs/R3L/`:

```python
# Core R3L workflow loop
def run(self):
    # 1. First rollout
    trajectory, reward, done, steps = first_rollout(self, env)

    # 2. Reflection (if not successful)
    if reward < 1.0:
        reflection = self.get_reflect(trajectory)
        pivot_point = reflection['retry_from_step']
        guidance = reflection['improvement_suggestion']

        # 3. Guided retry from pivot point
        second_trajectory, second_reward = second_rollout(
            self, env, guidance, trajectory, pivot_point
        )

        # 4. Adjust credit assignment
        self._adjust_action_mask_for_retry(exp, pivot_point)
```

## Data Format

Task data should be in JSONL format with the following structure:

```json
{"task_id": "unique_id", "task_desc": "task description or path"}
```

## Troubleshooting

### Common Issues

1. **OOM Error**: Reduce `gpu_memory_utilization` or `max_token_len_per_gpu`
2. **Slow Training**: Increase `engine_num` for more inference parallelism
3. **Environment Errors**: Ensure environment dependencies are installed (alfworld, scienceworld, etc.)

### Environment Installation

```bash
# ALFWorld
pip install alfworld

# ScienceWorld
pip install scienceworld

# WebShop (requires separate setup)
# See: https://github.com/princeton-nlp/WebShop
```
