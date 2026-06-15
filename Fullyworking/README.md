# Reinforcement Learning Based HVAC Control

A Soft Actor-Critic (SAC) agent trained to control a hydronic heat pump in a
single-zone residential building, using the
[BOPTEST](https://github.com/ibpsa/project1-boptest) high-fidelity building
emulator as the simulation environment.

The agent learns a price-aware, comfort-prioritising heating policy entirely
from simulation experience — no building model or manual tuning required.

---

## Table of Contents

- [Reinforcement Learning Based HVAC Control](#reinforcement-learning-based-hvac-control)
  - [Table of Contents](#table-of-contents)
  - [Background](#background)
  - [Project Structure](#project-structure)
  - [Environment Design](#environment-design)
    - [Simulation Environment](#simulation-environment)
    - [Observation Space](#observation-space)
    - [Action Space](#action-space)
    - [Comfort Schedule](#comfort-schedule)
  - [Reward Functions](#reward-functions)
    - [Hat-Shaped Reward](#hat-shaped-reward)
    - [Huber-Band Reward](#huber-band-reward)
    - [Potential-Based Reward Shaping (PBRS)](#potential-based-reward-shaping-pbrs)
  - [Agent Configuration](#agent-configuration)
  - [Baseline Controllers](#baseline-controllers)
  - [Setup \& Installation](#setup--installation)
    - [Prerequisites](#prerequisites)
    - [1 — Clone BOPTEST and start the Docker service](#1--clone-boptest-and-start-the-docker-service)
    - [2 — Clone this repository and install dependencies](#2--clone-this-repository-and-install-dependencies)
  - [Usage](#usage)
    - [Training](#training)
    - [Evaluation \& Visualisation](#evaluation--visualisation)
    - [Baseline Comparison](#baseline-comparison)
    - [RL Model Comparison](#rl-model-comparison)
  - [Results](#results)
    - [Controller Comparison](#controller-comparison)
    - [Reward Function Comparison](#reward-function-comparison)
    - [Scenario Testing](#scenario-testing)
  - [Discussion](#discussion)

---

## Background

Model Predictive Control (MPC) and classical PI controllers require accurate
building models and manual per-building tuning. This project investigates
whether a model-free RL agent can match or exceed classical controllers on
thermal comfort while simultaneously reducing energy cost under highly dynamic
electricity pricing.

The testcase is `bestest_hydronic_heat_pump` — a 48 m² single-zone apartment
with a hydronic heat pump, widely used in the academic HVAC control
literature as a realistic benchmark.

---

## Project Structure

```
BOPTEST_RL/
├── Fullyworking_2_reward/
│   ├── boptest_gym_env.py           # Custom Gymnasium environment + wrappers
│   ├── train_rl.py                  # Training script (SAC / PPO)
│   ├── requirements.txt             # Python dependencies
│   ├── Functionality_testing/
│   │   ├── interactive_env.py       # Manual step-through for debugging
│   │   └── validate_env.py          # Automated env sanity checks
│   └── Results_generator/
│       ├── episodic_visualisation.py  # Per-episode 3-page diagnostic plots
│       ├── academic_comparison.py     # RL vs Bang-Bang / PID / WC-PI
│       └── rl_comparison.py           # Side-by-side comparison of RL variants
└── trained_models/
    └── <run_id>/
        ├── config.json              # Hyperparameters & env config (committed)
        ├── final_model.zip          # Trained weights (gitignored — large)
        ├── training_progress.png    # Mean reward vs steps (gitignored)
        └── academic_comparison/     # Output from academic_comparison.py
```

---

## Environment Design

### Simulation Environment

BOPTEST is started as a Docker service and exposes a RESTful HTTP API.
Because BOPTEST does not conform to the Gymnasium interface, `boptest_gym_env.py`
provides a custom `BoptestGymEnv` wrapper that translates the API into standard
`reset()` / `step()` / `close()` calls.

Key episode parameters:

| Parameter | Value |
|---|---|
| Timestep | 1800 s (30 min) |
| Decisions per day | 48 |
| Episode length | 3 simulated days (259,200 s) |
| Warm-up period | 24 hours |
| Start time | Random (uniform over the year) |
| Parallel envs (training) | 8 × `SubprocVecEnv` |
| Effective training steps | 80,000 vectorised → 640,000 env interactions |

Random start times prevent overfitting to any single seasonal trajectory and
ensure the agent generalises across all seasons.

### Observation Space

All observations are min-max normalised to **[0, 1]** before being passed to
the policy network.

| Variable | Range | Description |
|---|---|---|
| `reaTZon_y` | 280 – 310 K | Current zone air temperature |
| `TDryBul` (4 steps) | 265 – 303 K | Outdoor dry-bulb temperature forecast at t+0, t+60, t+120, t+180 min |
| `HDirNor` (4 steps) | 0 – 862 W/m² | Direct normal solar irradiance forecast (same intervals) |
| `LowerSetp` (4 steps) | 280 – 310 K | Computed lower comfort setpoint forecast |
| `UpperSetp` (4 steps) | 280 – 310 K | Computed upper comfort setpoint forecast |
| ΔT_zone | −5.0 – +5.0 K/step | Zone temperature rate-of-change |
| sin(t), cos(t) | −1.0 – +1.0 | Circular time-of-day encoding (period = 86,400 s) |
| Last action | 0.0 – 1.0 | Previous heat pump modulation signal |

Forecasts use a 4-hour horizon at 60-minute intervals from the BOPTEST API.
Circular time encoding eliminates the midnight discontinuity that a plain
linear clock signal would create. The last-action feature gives the policy
memory of its previous decision, helping it distinguish self-induced
temperature changes from external disturbances.

### Action Space

A single continuous value in **[0, 1]**:
- **0** = heat pump fully off
- **1** = heat pump at maximum power

### Comfort Schedule

The agent must satisfy a time-varying comfort band that changes between occupied
and setback periods every day.

| Period | Hours | Temperature Range | Band Width |
|---|---|---|---|
| Day (occupied) | 07:00 – 22:00 | 21 – 24 °C | 3 °C |
| Night (setback) | 22:00 – 07:00 | 20 – 22 °C | 2 °C |

The agent must pre-heat before the morning occupied-period transition.
Electricity pricing follows BOPTEST's highly dynamic scenario, with prices
fluctuating significantly throughout the day.

---

## Reward Functions

Three reward formulations were implemented and compared. All combine a comfort
term with a price-weighted energy penalty. The comfort weight is set
substantially higher than the energy weight in every formulation, deliberately
prioritising thermal comfort over cost minimisation.

### Hat-Shaped Reward

Flat +1.0 inside the comfort band, linear penalty outside:

```
r_comfort = +1.0                           if T_lower ≤ T_zone ≤ T_upper
            −k · (T_lower − T_zone)        if T_zone < T_lower
            −k · (T_zone − T_upper)        if T_zone > T_upper

r_energy  = −ω · E · p(t)
r         = r_comfort + r_energy
```

where `k` is the linear penalty slope, `E` is energy consumed, and `p(t)` is
the dynamic electricity price. The flat in-band reward provides a constant
incentive but the discontinuity at the band edges can slow convergence.

### Huber-Band Reward

Zero reward inside the band, smooth quadratic-then-linear penalty outside:

```
e_c = max(0, T_lower − T_zone, T_zone − T_upper)

r_comfort = e_c² / (2δ)          if e_c ≤ δ
            e_c − δ/2             if e_c > δ

r_energy  = −ω · E · p(t)
```

Default weights: `w_c = 50`, `w_p = 30`, `δ = 0.25 K`.

### Potential-Based Reward Shaping (PBRS)

Policy-invariant shaping that provides a continuous gradient toward the band
centre (Ng, Harada & Russell, 1999):

```
Φ(s) = −(T_zone − T_center)²
F     = γ · Φ(s') − Φ(s)

r     = −w_e × (ΔE · p(t) / E_ref) + w_c × F
```

where `T_center = (T_lower + T_upper) / 2`, `w_c = 10`, `w_e = 1`,
`E_ref = 0.01 kWh/m²/step`. The PBRS γ must match SAC's γ (both 0.995).
Because `T_center` shifts with the schedule, PBRS naturally anticipates
morning/evening transitions.

---

## Agent Configuration

SAC was chosen over PPO and DQN because HVAC control requires continuous,
fine-grained modulation, and SAC is off-policy (it reuses all past transitions
via a replay buffer). DQN cannot handle a continuous action space. Each
environment step requires an HTTP round-trip to BOPTEST, making data collection
the dominant bottleneck — SAC's sample efficiency is therefore critical.

| Parameter | Value |
|---|---|
| Algorithm | SAC (Stable-Baselines3) |
| Network | 2 × 256 hidden layers, ReLU |
| Learning rate | 3 × 10⁻⁴ |
| Discount factor γ | 0.995 |
| Replay buffer | 100,000 transitions |
| Batch size | 256 |
| Entropy coefficient | auto |
| Parallel environments | 8 (SubprocVecEnv) |
| Vectorised / effective steps | 80,000 / 640,000 |

γ = 0.995 gives an effective horizon of ~200 steps (~100 hours), sufficient
for multi-day thermal reasoning and anticipatory pre-heating.

---

## Baseline Controllers

Three classical controllers are implemented in `academic_comparison.py` for
direct benchmarking under identical episodes and pricing conditions.

**Bang-Bang** — switches fully on below the lower comfort bound and fully off
above the upper bound. Lowest energy use but poorest comfort.

**PID** — PI controller tracking the comfort band midpoint. Modulation signal
clipped to [0, 1].

**WC-PI (SOTA)** — Weather-compensating PI controller with an outdoor
temperature feedforward term:

```
u(t) = clip(Kp · e(t) + I(t) + u_ff(t),  0,  1)
```

where `e(t) = T_centre − T_zone` and
`u_ff(t) = min(u_max, α · max(0, T_ref − T_cff))`.

The feedforward enables proactive pre-heating before cold spells reach the
zone. Anti-windup back-calculation (`T_i = T_i / 2`) prevents integral
wind-up during output saturation. Gains are SIMC-tuned for slow thermal
dynamics (`K_p = 0.5`, `T_i = 3600 s`).

---

## Setup & Installation

### Prerequisites

- Python 3.9+
- [Docker](https://docs.docker.com/get-docker/) with Docker Compose

### 1 — Clone BOPTEST and start the Docker service

```bash
git clone https://github.com/ibpsa/project1-boptest.git
cd project1-boptest
docker compose up web worker provision --scale worker=8
```

The API will be available at `http://127.0.0.1:8000` by default.
Leave this terminal running throughout training and evaluation.

### 2 — Clone this repository and install dependencies

```bash
git clone <this-repo-url>
cd BOPTEST_RL/Fullyworking_2_reward
pip install -r requirements.txt
```

The `requirements.txt` installs:
- `numpy`, `requests`, `matplotlib`
- `gymnasium`
- `stable-baselines3`

BOPTEST itself is an external Docker service and is **not** listed as a
pip dependency.

---

## Usage

All scripts are run from inside `Fullyworking_2_reward/`.

### Training

```bash
python train_rl.py
```

The script will:
1. Connect to the running BOPTEST Docker instance
2. Spin up 8 parallel SubprocVecEnv environments
3. Train a SAC agent for 80,000 vectorised steps
4. Save the model and training diagnostics under `../trained_models/<run_id>/`

The run directory is named automatically:
`SAC_comfort<wc>_price<we>_<N>envs_<timestamp>/`

**Output files** (per run):

| File | Description |
|---|---|
| `final_model.zip` | Trained SAC policy weights |
| `config.json` | Full hyperparameter & env config snapshot |
| `training_progress.png` | Mean reward per step over training |
| `kpi_history.pkl` | Raw KPI history for further analysis |

### Evaluation & Visualisation

```bash
python Results_generator/episodic_visualisation.py ../trained_models/<run_id>/
```

Produces three diagnostic figure pages per episode:

- **Page 1** — Zone temperature vs comfort band, outdoor temperature,
  discomfort penalty timeline
- **Page 2** — Heat pump control signal, total power breakdown,
  electricity price vs action (price-responsiveness check)
- **Page 3** — Per-step reward (stacked comfort + energy components),
  cumulative reward, policy scatter (action vs zone temperature)

### Baseline Comparison

Compare the trained SAC agent against Bang-Bang, PID, and WC-PI controllers
under identical episodes:

```bash
python Results_generator/academic_comparison.py ../trained_models/<run_id>/ \
    --episodes 2 \
    --url http://127.0.0.1:8000
```

Outputs to `<run_id>/academic_comparison/`:
- Per-episode 4-panel overlay plots
- KPI bar charts (energy, cost, thermal discomfort)
- `summary.txt` and `summary.csv` — tabular results suitable for a paper

### RL Model Comparison

Compare up to three trained RL variants side-by-side:

```bash
python Results_generator/rl_comparison.py \
    ../trained_models/<run_1>/ \
    ../trained_models/<run_2>/ \
    ../trained_models/<run_3>/ \
    --names "SAC-Hat,SAC-Huber,SAC-PBRS" \
    --episodes 2
```

---

## Results

Evaluation over 2 deterministic episodes (3 simulated days each) under highly
dynamic electricity pricing.

### Controller Comparison

| Controller | Energy (kWh) | Comfort Zone | Mean Temperature (°C) |
|---|---|---|---|
| **SAC (RL)** | **2487** | **84%** | **21.05** |
| WC-PI (SOTA) | 2441 | 81% | 21.85 |
| PID | 2601 | 71% | 21.81 |
| Bang-Bang | 2351 | 56% | 22.31 |

SAC achieves the highest thermal comfort (84%) while maintaining the lowest
mean zone temperature (21.05 °C). WC-PI delivers the second-best comfort
(81%) with the lowest energy draw. PID consumed the most energy (2601 kWh)
with only moderate comfort (71%). Bang-Bang used the least energy at the cost
of very poor thermal comfort (56%).

### Reward Function Comparison

| Reward Function | Energy (kWh/m²) | Comfort Zone | Mean Temperature (°C) |
|---|---|---|---|
| Hat-Shaped | 2487 | 83.6% | 21.05 |
| Huber-Band | 2655 | 81.2% | 21.72 |
| PBRS | 2639 | 86.3% | 21.69 |

- **Hat-shaped** is the most energy-efficient; the agent learns to hover near
  the lower comfort edge (no incentive to aim higher when already in-band).
- **Huber-band** leads to the highest energy consumption and the lowest comfort
  percentage; the zero in-band reward creates a policy that oscillates near the
  band edges.
- **PBRS** achieves the highest thermal comfort (86.3%) at moderate energy
  cost; the continuous gradient toward the band centre naturally anticipates
  schedule transitions.

### Scenario Testing

The best model (SAC + PBRS) was tested across three out-of-sample scenarios:

| Scenario | Comfort Zone | Notes |
|---|---|---|
| Normal (winter) | 83.6% | Stable tracking near lower comfort bound |
| Extreme cold (< −12 °C outdoor) | 86.5% | Minor violations only under extreme cold snaps |
| High energy pricing (×2 multiplier) | 45.7% | Agent trades comfort for cost when pricing dominates |

---

## Discussion

**Why RL outperforms classical controllers on comfort:**  
SAC learns a schedule-aware modulating policy from experience. Classical
controllers are reactive; SAC can pre-heat before a cold morning or a schedule
transition, reducing the need for high-power recovery heating.

**Why WC-PI uses less energy despite lower comfort:**  
The feedforward compensates for outdoor temperature but WC-PI cannot infer
the electricity price signal. SAC reduces heating specifically during price
peaks, achieving a better energy–cost trade-off rather than simply minimising
kWh.

**Why simulation-based training:**  
RL requires hundreds of thousands of environment interactions including
sub-optimal exploratory actions — unsafe and impractical on real hardware.
BOPTEST simulates a 3-day episode in seconds. The long-term vision is
pre-training in simulation followed by fine-tuning on real building data.

**Limitations:**  
- Trained and evaluated on a single testcase (`bestest_hydronic_heat_pump`).
- Performance degrades under extreme conditions (outdoor temperature < −12 °C).
- The agent has no explicit model of the building; generalisation to different
  building archetypes requires retraining or transfer learning.
