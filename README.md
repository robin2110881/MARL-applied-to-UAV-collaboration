# MARL-applied-to-UAV-collaboration

Implementation of single-agent and multi-agent reinforcement learning for UAV payload collaboration using PyBullet drone simulations and Proximal Policy Optimization (PPO).

---

# Installation

Tested with:

```text
Python 3.11.9
```

## 1. Install dependencies

Create and activate a virtual environment, then install the required packages:

```bash
pip install -r requirements.txt
```

## 2. Install the PyBullet simulator

Navigate to the simulator folder:

```bash
cd gym-pybullet-drones
```

Install the package in editable mode:

```bash
pip install -e .
```

---

# Running Experiments

## Single-Agent Training

```bash
python SingleAgent/PPO_Learning_SA_hover.py
```

## Multi-Agent Training

```bash
python MultiAgent/PPO_Learning_MA_hover.py
```

---

# Testing

The trained policies can be evaluated using the provided testing scripts. Two versions are available depending on the training approach: single-agent PPO and multi-agent PPO (MAPPO).

## Single-Agent PPO

Run:

```bash
python SingleAgent/PPO_Testing_SA_hover.py
```

## Multi-Agent PPO (MAPPO)

Run:

```bash
python MultiAgent/PPO_Testing_MA_hover.py
```

## Checkpoint Files

The testing scripts automatically load trained checkpoints from the following directories:

```
SingleAgent/sa_ppo/
```

for single-agent PPO, and:

```
MultiAgent/ma_ppo/
```

for MAPPO.

Each checkpoint consists of the following files:

```
actor_torch_ppo
critic_torch_ppo
scaler_torch_ppo.pkl
```

The repository includes default trained weights that can be directly tested.

To evaluate a different trained model, replace the existing checkpoint files with the desired weights. The new files must be renamed as:

```
actor_torch_ppo
critic_torch_ppo
scaler_torch_ppo.pkl
```

Before testing, ensure that the actor network architecture defined in the testing script matches the architecture used during training. If a different network size was used, update the network dimensions accordingly before loading the checkpoint.

---

# Project Structure

```text
MARL-applied-to-UAV-collaboration/
│
├── gym-pybullet-drones/     # PyBullet simulator and custom environments
├── SingleAgent/             # Single-agent PPO training/testing
├── MultiAgent/              # Multi-agent PPO training/testing
└── Demo Files/              # Control and parameter demos
```

---

# Environment Files

Located in:

```text
gym-pybullet-drones/gym_pybullet_drones/
```

Main files:

- `envs/BaseAviary.py`  
  Base PyBullet simulation and payload interaction logic.

- `envs/SA_Aviary.py`  
  Single-agent UAV environment.

- `envs/MA_Aviary.py`  
  Multi-agent UAV collaboration environment.

- `envs/Parallel_SA_Aviary.py`  
  Parallel single-agent environment wrapper.

- `envs/Parallel_MA_Aviary.py`  
  Parallel multi-agent environment wrapper.

- `control/BaseControl.py`  
  Base drone control interface.

- `control/ForceControl.py`  
  Force-based payload controller.

---

# PPO Scripts

## Single-Agent

Located in:

```text
SingleAgent/
```

Training:

- `PPO_Learning_SA.py`
- `PPO_Learning_SA_hover.py`

Testing:

- `PPO_Testing_SA_hover.py`

---

## Multi-Agent

Located in:

```text
MultiAgent/
```

Training:

- `PPO_Learning_MA.py`
- `PPO_Learning_MA_hover.py`

Testing:

- `PPO_Testing_MA_hover.py`

---

# Outputs

Training checkpoints and results are saved in:

```text
SingleAgent/
├── sa_ppo/
└── sa_ppo_training_outputs/

MultiAgent/
├── ma_ppo/
└── ma_ppo_training_outputs/
```

Typical output files:

```text
actor_torch_ppo
critic_torch_ppo
scaler_torch_ppo.pkl
score_history_*.npy
length_history_*.npy
value_loss_history_*.npy
entropy_loss_history_*.npy
```

---

# Demo Files

Located in:

```text
Demo Files/
```

Available scripts:

- `demo_force_control.py`  
  Basic force-control demonstration.

- `demo_force_control_parrelel.py`  
  Parallel force-control demonstration.

- `find_rope_parameter.py`  
  Rope parameter tuning utility.

---