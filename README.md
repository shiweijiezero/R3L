<div align="center">
  <h1>R<sup>3</sup>L</h1>
  <p><strong>Reflect-then-Retry Reinforcement Learning</strong></p>
  <p>Language-Guided Exploration, Pivotal Credit and Positive Amplification</p>
</div>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
</p>

---

R³L is a reinforcement learning algorithm that improves LLM reasoning and agentic capabilities through language-guided exploration, pivotal credit assignment, and positive amplification. Built on [Trinity-RFT](https://github.com/modelscope/Trinity-RFT).

## Installation

```bash
conda create -n r3l python=3.10
conda activate r3l

pip install -e ".[dev]"
pip install flash-attn==2.8.1 --no-build-isolation
```

## Quick Start

```bash
# 1. Set environment variables
export TRINITY_MODEL_PATH=/path/to/Qwen2.5-1.5B-Instruct
export TRINITY_CHECKPOINT_ROOT_DIR=./checkpoints

# 2. Start Ray cluster
ray start --head

# 3. Run training
# ALFWorld
trinity run --config examples/R3L/alfworld/opmd_R3L_1.5B.yaml

# WebShop
trinity run --config examples/R3L/webshop/opmd_R3L_1.5B.yaml

# ScienceWorld
trinity run --config examples/R3L/scienceworld/opmd_R3L_1.5B.yaml

# DAPO (Math)
trinity run --config examples/R3L/dapo/opmd_R3L_1.5B.yaml
```

## Supported Environments

| Environment | Config | Notes |
|-------------|--------|-------|
| ALFWorld | `examples/R3L/alfworld/opmd_R3L_*.yaml` | Requires `alfworld` package |
| WebShop | `examples/R3L/webshop/opmd_R3L_*.yaml` | Requires ~1TB memory |
| ScienceWorld | `examples/R3L/scienceworld/opmd_R3L_*.yaml` | Requires Java |
| DAPO | `examples/R3L/dapo/opmd_R3L_*.yaml` | Auto-download from HuggingFace |

Model sizes: `1.5B`, `3B`, `7B`

## Baselines

| Method | Config Pattern |
|--------|----------------|
| GRPO | `examples/R3L/<env>/grpo_*.yaml` |
| OPMD | `examples/R3L/<env>/opmd_*.yaml` |
| RAFT | `examples/R3L/<env>/RAFT_*.yaml` |
| Reflect-GRPO | `examples/R3L/<env>/reflect_grpo_*.yaml` |
| Critique-GRPO | `examples/R3L/<env>/critique_grpo_*.yaml` |

## Ablation Variants

| Variant | Config Pattern |
|---------|----------------|
| R³L (Full) | `examples/R3L/<env>/opmd_R3L_*.yaml` |
| w/o Pivotal Credit | `examples/R3L/<env>/opmd_R3L_w_o_credit_*.yaml` |
| w/o Positive Amplification | `examples/R3L/<env>/opmd_R3L_w_o_reweight_*.yaml` |
| w/o Reflect-Retry | `examples/R3L/<env>/opmd_reweight_adv_*.yaml` |

## Configuration

Key hyperparameters:

```yaml
algorithm:
  algorithm_type: opmd_reweight_adv  # R3L algorithm
  repeat_times: 8                     # Trajectories per task

model:
  model_path: ${oc.env:TRINITY_MODEL_PATH}
  max_response_tokens: 512

buffer:
  explorer_input:
    default_workflow_type: 'R3L_alfworld_workflow'
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `repeat_times` | 8 | Group size (N) |
| `α` | 3.0 | Positive amplification factor |
| `lr` | 1e-6 | Learning rate |
| `temperature` | 1.0 | Training temperature |

## Project Structure

```
R3L/
├── trinity/common/workflows/envs/R3L/   # R3L workflows
│   ├── alfworld/
│   ├── webshop/
│   ├── scienceworld/
│   └── dapo/
├── trinity/algorithm/                    # Algorithm implementations
│   └── advantage_fn/opmd_advantage.py   # Positive Amplification
└── examples/R3L/                         # Training configs
```

## Environment Setup

### ALFWorld

**Step 1: Install alfworld**

```bash
pip install alfworld
```

**Step 2: Download data**

```bash
# Option 1: Auto download to ~/.cache/alfworld/
alfworld-download

# Option 2: Specify download path
alfworld-download --data-dir ./alf-data
```

**Step 3: Configure data path**

Edit `examples/R3L/alfworld/get_alfworld_data.py`:

```python
# Line 11, modify to your actual data path
alfworld_data_root = "/your/local/path/alfworld/json_2.1.1"
```

> **Note**: Keep `json_2.1.1` at the end of the path.

**Step 4: Process data**

```bash
cd examples/R3L/alfworld
python get_alfworld_data.py
```

Processed data will be saved to `examples/R3L/alfworld/alfworld_data/`.

**Step 5: Start training**

```bash
trinity run --config examples/R3L/alfworld/opmd_R3L_1.5B.yaml
```

### WebShop

> **Note**: WebShop requires ~1TB memory. Skip if resources are limited.

**Step 1: Clone WebShop repository**

```bash
git clone https://github.com/princeton-nlp/webshop.git webshop
cd webshop
```

**Step 2: Install Java 17+**

```bash
# Using conda
conda install -c conda-forge openjdk=17
```

**Step 3: Run setup script**

```bash
# Small dataset (recommended for testing)
./setup.sh -d small

# Full dataset
./setup.sh -d all
```

Follow WebShop's installation instructions. Note that some Python dependencies may conflict with R3L - install them individually if needed.

**Step 4: Process data**

```bash
cd examples/R3L/webshop
python get_webshop_data.py
```

**Step 5: Configure WebShop path**

Option A: Set environment variable

```bash
export WEBSHOP_PATH=/path/to/webshop
```

Option B: Modify workflow files directly

Edit path in all WebShop workflow files (`trinity/common/workflows/envs/R3L/webshop/*.py`):

```python
# Find this line and update the path
sys.path.append("/your/path/to/webshop")
```

**Step 6: Start training**

```bash
trinity run --config examples/R3L/webshop/opmd_R3L_1.5B.yaml
```

### ScienceWorld

**Step 1: Clone and install ScienceWorld**

```bash
git clone https://github.com/allenai/ScienceWorld.git
cd ScienceWorld
pip install .
```

**Step 2: Configure jar path**

Edit `examples/R3L/scienceworld/get_sciworld_data.py`:

```python
# Line 101-103, set the jar path to your ScienceWorld directory
jar_path = "/your/path/ScienceWorld/scienceworld/scienceworld.jar"
```

**Step 3: Process data**

```bash
cd examples/R3L/scienceworld
python get_sciworld_data.py
```

**Step 4: Start training**

```bash
trinity run --config examples/R3L/scienceworld/opmd_R3L_1.5B.yaml
```

### DAPO

DAPO uses HuggingFace datasets that are automatically downloaded. No manual setup required.

```bash
trinity run --config examples/R3L/dapo/opmd_R3L_1.5B.yaml
```

## License

Apache License 2.0

## Acknowledgements

- [Trinity-RFT](https://github.com/modelscope/Trinity-RFT)
- [ALFWorld](https://github.com/alfworld/alfworld), [ScienceWorld](https://github.com/allenai/ScienceWorld), [WebShop](https://github.com/princeton-nlp/WebShop)
- [DAPO](https://github.com/BytedTsinghua-SIA/DAPO)
