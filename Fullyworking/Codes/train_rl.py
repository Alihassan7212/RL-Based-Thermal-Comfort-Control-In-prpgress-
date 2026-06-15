"""
Training script for BOPTEST RL using improved BoptestGymEnv
"""

import sys
import os
import signal
import json
import pickle
import datetime
import traceback
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — safe for headless/server use
import matplotlib.pyplot as plt
try:
    from stable_baselines3 import DQN, PPO, SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
except ModuleNotFoundError:
    DQN = PPO = SAC = None
    class BaseCallback:  # minimal shim so module can be imported without SB3
        def __init__(self, verbose=0):
            self.verbose = verbose
            self.n_calls = 0
            self.training_env = None
            self.model = None
    Monitor = None
    SubprocVecEnv = DummyVecEnv = None

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from boptest_gym_env import (
    BoptestGymEnv,
    NormalizedObservationWrapper,
    DiscretizedActionWrapper,
    CustomRewardWrapper,
    PBRSRewardWrapper,
    ComputedSetpointObsWrapper,
    ZoneTempDeltaWrapper,
    SinCosTimeWrapper,
    LastActionWrapper,
)

# Global variable to track environment for cleanup
_global_env = None


def _require_sb3():
    """Raise a clear error when Stable-Baselines3 is not available."""
    if any(v is None for v in (DQN, PPO, SAC, Monitor, SubprocVecEnv, DummyVecEnv)):
        raise RuntimeError(
            "stable_baselines3 is required for training/evaluation scripts. "
            "Install it in the active Python environment."
        )


def cleanup_handler(signum, frame):
    """Handle interruption and clean up environment."""
    global _global_env
    print("\n\nInterrupted! Cleaning up BOPTEST...")
    if _global_env is not None:
        try:
            _global_env.close()
            print("✓ Environment closed successfully")
        except Exception as e:
            print(f"⚠ Error during cleanup: {e}")
    print("\nIf BOPTEST is still stuck, run: python cleanup_boptest.py")
    sys.exit(0)


# Register signal handlers for clean interruption
signal.signal(signal.SIGINT, cleanup_handler)
signal.signal(signal.SIGTERM, cleanup_handler)


def _find_wrapper(env, cls):
    """Walk the wrapper stack and return the first instance of ``cls``.

    Returns ``None`` if not found.
    """
    current = env
    while current is not None:
        if isinstance(current, cls):
            return current
        current = getattr(current, 'env', None)
    return None


def _plot_training_progress(kpi_history, save_path):
    """Save a training progress graph (mean reward vs steps) to save_path."""
    if not kpi_history:
        return
    steps   = [e['step'] for e in kpi_history]
    rewards = [e.get('mean_reward', 0) for e in kpi_history]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, rewards, linewidth=1.5, color='steelblue')
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Training steps')
    ax.set_ylabel('Mean reward (per step)')
    ax.set_title('Training progress — mean reward per step')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)


class KPILoggingCallback(BaseCallback):
    """Callback to log KPIs during training."""

    def __init__(self, eval_freq=1000, verbose=1,
                 save_dir=None, checkpoint_freq=50000):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.kpi_history = []
        self._reward_since_last = 0.0   # accumulates rewards between KPI prints
        self.save_dir = save_dir
        self.checkpoint_freq = checkpoint_freq

    def _on_step(self):
        # Accumulate step reward(s) every call so we can report mean reward
        step_rewards = self.locals.get('rewards', np.array([0.0]))
        self._reward_since_last += float(np.sum(step_rewards))

        # ── KPI log every eval_freq steps ────────────────────────────────────
        if self.n_calls % self.eval_freq == 0:
            try:
                # n_envs: number of parallel environments (1 for single env)
                n_envs = getattr(self.training_env, 'num_envs', 1)

                # VecEnv (SubprocVecEnv / DummyVecEnv): use env_method so the
                # call executes inside the subprocess where the env lives.
                # Single env: fall back to _find_wrapper traversal.
                if hasattr(self.training_env, 'env_method'):
                    kpis_list = self.training_env.env_method('get_kpis')
                    # Filter out None entries (env failed to return KPIs)
                    valid_kpis = [e for e in kpis_list if e is not None]
                    # Average KPIs across all parallel envs; guard against None
                    # values within a dict (KPIs not yet computed return null).
                    if valid_kpis:
                        kpis = {
                            k: float(np.mean([
                                float(e.get(k) if e.get(k) is not None else 0.0)
                                for e in valid_kpis
                            ]))
                            for k in valid_kpis[0]
                        }
                    else:
                        kpis = {}
                else:
                    env = _find_wrapper(self.training_env, BoptestGymEnv)
                    if env is None:
                        return True
                    kpis = env.get_kpis()

                # Divide by (eval_freq × n_envs): n_calls counts vectorised
                # steps, each of which advances n_envs individual episodes.
                mean_reward = self._reward_since_last / (self.eval_freq * n_envs)
                self.kpi_history.append({
                    'step': self.n_calls,
                    'kpis': kpis,
                    'mean_reward': mean_reward,
                })
                self._reward_since_last = 0.0   # reset for next interval

                env_label = f"{n_envs} envs" if n_envs > 1 else "1 env"
                if self.verbose:
                    print(f"\n=== Training step {self.n_calls} ({env_label}) ===")
                    print(f"Energy:      {kpis.get('ener_tot', 0):.4f} kWh/m²")
                    print(f"Discomfort:  {kpis.get('tdis_tot', 0):.4f} K·h/m²")
                    print(f"Cost:        {kpis.get('cost_tot', 0):.4f} EUR/m²")
                    print(f"Mean reward: {mean_reward:.3f}  (last {self.eval_freq} steps/env)")
            except Exception as e:
                if self.verbose:
                    print(f"Failed to get KPIs: {e}")

        # ── Checkpoint + graph every checkpoint_freq steps ───────────────────
        if self.save_dir and self.n_calls > 0 and self.n_calls % self.checkpoint_freq == 0:
            try:
                ckpt_name = f"checkpoint_step_{self.n_calls:07d}"
                ckpt_path = os.path.join(self.save_dir, ckpt_name)
                self.model.save(ckpt_path)
                if self.verbose:
                    print(f"\n[Checkpoint] step {self.n_calls:,} → {ckpt_name}.zip")
            except Exception as e:
                if self.verbose:
                    print(f"\n[Checkpoint] step {self.n_calls:,} — save failed: {e}")
            try:
                graph_path = os.path.join(self.save_dir, "training_progress.png")
                _plot_training_progress(self.kpi_history, graph_path)
                if self.verbose:
                    print(f"[Graph]      training_progress.png updated")
            except Exception as e:
                if self.verbose:
                    print(f"[Graph]      training_progress.png — plot failed: {e}")

        return True


def create_env(
    url='http://127.0.0.1:8000',
    testcase='bestest_hydronic_heat_pump',
    use_custom_reward=True,
    comfort_weight=50.0,
    price_weight=30.0,
    max_pen=20.0,
    cliff_reward=True,          # True = cliff at band edge; False = legacy continuous Huber
    use_discrete_actions=False,
    n_bins_act=10,
    env_params=None,            # pass saved env_params dict to recreate training env exactly
    comfort_schedule=None,      # None → CustomRewardWrapper uses DEFAULT_COMFORT_SCHEDULE
    use_zone_temp_delta=True,   # append ΔT_zone (K/step) to observations
    predictive_period=None,     # None → use default (4 * predictive_step); overridden by env_params
    predictive_step=None,       # None → use default (2 * step_period = 60 min); overridden by env_params
    regressive_period=None,     # None → no historical obs; overridden by env_params
    use_time_observation=False, # append simulation time in observation vector
    random_start_time=False,    # False by default for reproducible training/evaluation
    start_time=0,               # fixed start when random_start_time=False
    max_price_per_step=0.01,    # normalisation for price reward (EUR/m²/step)
    constant_price=0.1,         # fixed electricity price (EUR/kWh)
    reward_type='pbrs',         # 'pbrs' = PBRSRewardWrapper; 'huber' = CustomRewardWrapper
    pbrs_gamma=0.995,           # PBRS: discount factor (MUST match SAC gamma exactly)
    pbrs_comfort_weight=10.0,   # PBRS: shaping term scale (w_c)
    pbrs_energy_weight=1.0,     # PBRS: energy penalty scale (w_e)
    pbrs_E_ref=0.01,            # PBRS: kWh/m²/step energy normalisation constant
    _register_global=True,      # set False when creating envs for a VecEnv
):
    """Create and wrap BOPTEST environment.

    Pass env_params (loaded from config.json) to recreate the exact environment
    used during training — critical for visualization of saved models.
    """

    global _global_env

    # Defaults — direct kwargs take priority; env_params can further override
    step_period        = 1800       # 30-minute steps
    if predictive_step is None:
        predictive_step    = 2 * step_period   # 3600 s – 60-min forecast interval (2× step_period)
    if predictive_period is None:
        predictive_period  = 4 * predictive_step   # 14400 s → t+0, t+60 min, t+120 min, t+180 min
    if regressive_period is None:
        regressive_period  = None       # no historical observations
    max_episode_length = 3 * 24 * 3600  # 3-day episodes
    warmup_period      = 24 * 3600
    comfort_temp_range = (294.15, 297.15)   # 21-24°C (day default, 3°C band)

    # Observations for bestest_hydronic_heat_pump:
    #   reaTZon_y          — current zone temperature (measurement)
    #   TDryBul            — outdoor dry-bulb temperature forecast (4 steps)
    #   HDirNor            — direct normal solar irradiance forecast (4 steps)
    #   LowerSetp/UpperSetp are injected analytically by ComputedSetpointObsWrapper (4 steps each)
    #   ZoneTempDelta      — zone temperature rate-of-change (K/step), injected by ZoneTempDeltaWrapper
    #   sin/cos time       — circular time-of-day encoding, injected by SinCosTimeWrapper
    #                        (replaces raw time: no discontinuity at midnight)
    observations = {
        'reaTZon_y': (280.,  310.),
        'TDryBul':   (265,   303),
        'HDirNor':   (0,     862),
    }
    # use_time_observation=True → SinCosTimeWrapper added in the stack below.
    # Raw 'time' is intentionally excluded from observations_dict so the base
    # env never appends raw time values (5 redundant slots).

    # bestest_hydronic_heat_pump does not use time_period
    scenario = {
        'electricity_price': 'constant',
    }

    if env_params is not None:
        predictive_period  = env_params.get('predictive_period',  predictive_period)
        # Read step_period first so backward-compat logic below can use it.
        step_period        = env_params.get('step_period',        step_period)

        # ── Backward compatibility: old configs (pre-SinCosTimeWrapper) ───────
        # Old models stored raw 'time' as a base-env observation and had no
        # explicit predictive_step key (they used step_period as the interval).
        # Detect this by checking for 'time' in observations and absence of
        # 'predictive_step'.  For these models we must:
        #   1. Keep 'time' in the observations dict (raw time, not sin/cos).
        #   2. Use step_period as predictive_step (old 1-to-1 default).
        #   3. NOT add SinCosTimeWrapper (raw time already covers this).
        _is_old_raw_time_config = (
            'predictive_step' not in env_params
            and 'time' in env_params.get('observations', {})
        )

        if _is_old_raw_time_config:
            predictive_step = step_period   # old default: interval == step
        else:
            predictive_step = env_params.get('predictive_step', predictive_step)

        regressive_period  = env_params.get('regressive_period',  regressive_period)
        max_episode_length = env_params.get('max_episode_length', max_episode_length)
        warmup_period      = env_params.get('warmup_period',      warmup_period)
        random_start_time  = env_params.get('random_start_time',  random_start_time)
        start_time         = env_params.get('start_time',         start_time)
        use_discrete_actions = env_params.get('use_discrete_actions', use_discrete_actions)
        if 'comfort_temp_range' in env_params:
            comfort_temp_range = tuple(env_params['comfort_temp_range'])
        if 'observations' in env_params:
            if _is_old_raw_time_config:
                # Old model: preserve 'time' as a raw measurement/forecast variable.
                observations = {k: tuple(v) for k, v in env_params['observations'].items()}
            else:
                # New model: strip 'time' — handled by SinCosTimeWrapper below.
                observations = {k: tuple(v) for k, v in env_params['observations'].items()
                                if k != 'time'}
        if 'use_time_observation' in env_params:
            use_time_observation = bool(env_params['use_time_observation'])
            if _is_old_raw_time_config:
                # Old models embedded raw time in the base env; SinCosTimeWrapper
                # must NOT be added or the observation dimension will be wrong.
                use_time_observation = False
        if 'scenario' in env_params:
            scenario = env_params['scenario']
        if 'comfort_schedule' in env_params:
            comfort_schedule = [
                (e[0], e[1], tuple(e[2])) for e in env_params['comfort_schedule']
            ]
        if 'comfort_weight' in env_params:
            comfort_weight = env_params['comfort_weight']
        if 'price_weight' in env_params:
            price_weight = env_params['price_weight']
        elif 'power_weight' in env_params:
            price_weight = env_params['power_weight']   # backward-compat with old configs
        if 'max_pen' in env_params:
            max_pen = env_params['max_pen']
        if 'cliff_reward' in env_params:
            cliff_reward = env_params['cliff_reward']
        if 'max_price_per_step' in env_params:
            max_price_per_step = env_params['max_price_per_step']
        if 'constant_price' in env_params:
            constant_price = env_params['constant_price']
        # PBRS params — old configs without these keys keep the kwarg defaults
        if 'reward_type' in env_params:
            reward_type = env_params['reward_type']
        pbrs_gamma          = env_params.get('pbrs_gamma',          pbrs_gamma)
        pbrs_comfort_weight = env_params.get('pbrs_comfort_weight', pbrs_comfort_weight)
        pbrs_energy_weight  = env_params.get('pbrs_energy_weight',  pbrs_energy_weight)
        pbrs_E_ref          = env_params.get('pbrs_E_ref',          pbrs_E_ref)
        # Old configs without this key default to False (no wrapper present in old model)
        use_zone_temp_delta = env_params.get('use_zone_temp_delta', False)

    # Create base environment
    env = BoptestGymEnv(
        url=url,
        testcase=testcase,
        actions=['oveHeaPumY_u'],
        observations=observations,
        predictive_period=predictive_period,
        predictive_step=predictive_step,
        regressive_period=regressive_period,
        random_start_time=random_start_time,
        start_time=start_time,
        max_episode_length=max_episode_length,
        warmup_period=warmup_period,
        step_period=step_period,
        scenario=scenario,
    )

    # Inject analytically-computed LowerSetp / UpperSetp setpoint forecast.
    # bestest_hydronic_heat_pump does not expose these as BOPTEST forecast points,
    # so we derive them from the comfort schedule directly.  Must sit below
    # NormalizedObservationWrapper so that the setpoints are normalised like all
    # other observations (setp_bounds = (280, 310) K → normalised to [0, 1]).
    env = ComputedSetpointObsWrapper(
        env,
        comfort_schedule=comfort_schedule,
        comfort_temp_range=comfort_temp_range,
        predictive_period=predictive_period,
        step_period=step_period,
        predictive_step=predictive_step,
    )

    # Append zone temperature rate-of-change (ΔT, K/step) before normalisation
    # so it is scaled to [0, 1] along with all other raw observations.
    # Omitted for old saved models that were trained without this feature.
    if use_zone_temp_delta:
        env = ZoneTempDeltaWrapper(env)

    # Append sin/cos circular time-of-day encoding before normalisation.
    # Replaces 5 raw time slots (4 forecast steps + 1 duplicate current
    # measurement) with 2 continuous, periodic features.  Bounds [−1, +1]
    # are mapped to [0, 1] by NormalizedObservationWrapper below.
    if use_time_observation:
        env = SinCosTimeWrapper(env)

    # Apply normalization (covers all obs including setpoints, ΔT and sin/cos)
    env = NormalizedObservationWrapper(env)

    # Apply reward wrapper
    if use_custom_reward:
        if reward_type == 'pbrs':
            # Potential-Based Reward Shaping (Ng et al. 1999) — policy-invariant,
            # dense inside-band gradient, schedule-anticipatory via T_center.
            # pbrs_gamma MUST equal the gamma passed to train_sac().
            env = PBRSRewardWrapper(
                env,
                gamma=pbrs_gamma,
                comfort_weight=pbrs_comfort_weight,
                energy_weight=pbrs_energy_weight,
                E_ref=pbrs_E_ref,
                comfort_schedule=comfort_schedule,
                comfort_temp_range=comfort_temp_range,
            )
        else:
            # Huber-band reward (backward-compatible fallback)
            env = CustomRewardWrapper(
                env,
                comfort_weight=comfort_weight,
                price_weight=price_weight,
                max_pen=max_pen,
                comfort_temp_range=comfort_temp_range,
                comfort_schedule=comfort_schedule,
                cliff_reward=cliff_reward,
                max_price_per_step=max_price_per_step,
                constant_price=constant_price,
            )

    # Append last action to observations (helps agent reason about thermal inertia)
    env = LastActionWrapper(env)

    # Apply discretization if requested
    if use_discrete_actions:
        env = DiscretizedActionWrapper(env, n_bins_act=n_bins_act)

    # Wrap with Monitor for tracking
    if Monitor is not None:
        env = Monitor(env)

    # Track for cleanup (skip when building envs for a VecEnv — the VecEnv
    # itself will be registered in _global_env by make_boptest_vec_env).
    if _register_global:
        _global_env = env

    return env


def make_boptest_vec_env(
    n_envs=4,
    url='http://127.0.0.1:8000',
    use_subprocess=True,
    **create_env_kwargs,
):
    """Create a vectorized BOPTEST environment with *n_envs* parallel instances.

    Each instance registers its own testid with the BOPTEST server, so they
    run completely independently.  The server must be started with at least
    *n_envs* workers::

        docker compose up web worker provision --scale worker=<n_envs>

    Parameters
    ----------
    n_envs : int
        Number of parallel environments.  Must not exceed the number of
        BOPTEST workers available on the server.
    url : str
        BOPTEST REST API URL shared by all instances.
    use_subprocess : bool
        ``True``  → :class:`SubprocVecEnv` (recommended; each env runs in its
                    own process so HTTP calls happen truly in parallel).
        ``False`` → :class:`DummyVecEnv`  (sequential; easier to debug).
    **create_env_kwargs
        Any keyword argument accepted by :func:`create_env` (e.g.
        ``comfort_weight``, ``price_weight``, ``use_custom_reward``).
        ``_register_global`` is always forced to ``False`` for sub-envs.
    """
    _require_sb3()
    global _global_env

    # Force _register_global=False so each create_env call doesn't overwrite
    # _global_env — the VecEnv is registered below instead.
    create_env_kwargs.pop('_register_global', None)

    def _make_env():
        def _init():
            return create_env(url=url, _register_global=False, **create_env_kwargs)
        return _init

    env_fns = [_make_env() for _ in range(n_envs)]

    if use_subprocess:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    # Register the VecEnv for cleanup by the signal handler
    _global_env = vec_env
    return vec_env


def train_dqn(
    env,
    total_timesteps=5000,
    learning_rate=5e-4,
    batch_size=24,
    buffer_size=365 * 24,
    gamma=0.99,
    verbose=1
):
    """Train DQN agent."""
    _require_sb3()

    model = DQN(
        'MlpPolicy',
        env,
        verbose=verbose,
        gamma=gamma,
        learning_rate=learning_rate,
        batch_size=batch_size,
        buffer_size=buffer_size,
        learning_starts=batch_size,
        train_freq=1,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_final_eps=0.05,
        seed=123456
    )

    callback = KPILoggingCallback(eval_freq=1000, verbose=verbose)
    model.learn(total_timesteps=total_timesteps, callback=callback)

    return model, callback.kpi_history


def train_ppo(
    env,
    total_timesteps=5000,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    verbose=1,
    save_dir=None,
    checkpoint_freq=50000,
):
    """Train PPO agent (for continuous actions)."""
    _require_sb3()

    model = PPO(
        'MlpPolicy',
        env,
        verbose=verbose,
        gamma=gamma,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        ent_coef=0.05,   # S3: entropy bonus prevents premature convergence to constant action
        seed=123456
    )

    callback = KPILoggingCallback(
        eval_freq=1000, verbose=verbose,
        save_dir=save_dir, checkpoint_freq=checkpoint_freq,
    )
    model.learn(total_timesteps=total_timesteps, callback=callback)

    return model, callback.kpi_history


def train_sac(
    env,
    total_timesteps=50000,
    learning_rate=3e-4,
    buffer_size=100000,
    batch_size=256,
    gamma=0.995,
    verbose=1,
    save_dir=None,
    checkpoint_freq=50000,
):
    """Train SAC agent (for continuous actions)."""
    _require_sb3()

    model = SAC(
        'MlpPolicy',
        env,
        verbose=verbose,
        gamma=gamma,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        learning_starts=batch_size,
        ent_coef='auto',   # automatic entropy tuning
        seed=123456
    )

    callback = KPILoggingCallback(
        eval_freq=1000, verbose=verbose,
        save_dir=save_dir, checkpoint_freq=checkpoint_freq,
    )
    model.learn(total_timesteps=total_timesteps, callback=callback)

    return model, callback.kpi_history


def evaluate_agent(env, model, n_episodes=5):
    """Evaluate trained agent."""

    episode_rewards = []
    episode_kpis = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        eval_env = _find_wrapper(env, BoptestGymEnv)
        kpis = eval_env.get_kpis() if eval_env else {}
        episode_rewards.append(episode_reward)
        episode_kpis.append(kpis)

        print(f"\nEpisode {ep + 1}/{n_episodes}:")
        print(f"  Total reward: {episode_reward:.3f}")
        print(f"  Energy: {kpis.get('ener_tot', 0):.4f}")
        print(f"  Discomfort: {kpis.get('tdis_tot', 0):.4f}")
        print(f"  Cost: {kpis.get('cost_tot', 0):.4f}")
        print(f"  Emissions: {kpis.get('emis_tot', 0):.4f}")

    return episode_rewards, episode_kpis


if __name__ == "__main__":
    _require_sb3()

    URL      = 'http://127.0.0.1:8000'
    TESTCASE = 'bestest_hydronic_heat_pump'
    N_ENVS   = 8   # parallel BOPTEST workers; server must be started with --scale worker=N_ENVS

    # ── Training hyper-parameters ─────────────────────────────────────────────
    algorithm            = "SAC"
    _SAC_GAMMA           = 0.995  # SAC discount factor — governs bootstrapping horizon

    # ── PBRS reward parameters ────────────────────────────────────────────────
    # pbrs_gamma MUST equal _SAC_GAMMA.  Any mismatch weakens the Ng 1999
    # policy-invariance guarantee (F = γΦ(s') − Φ(s) requires the same γ).
    pbrs_gamma_v         = _SAC_GAMMA  # 0.995
    pbrs_comfort_w       = 10.0        # w_c: shaping term scale
    pbrs_energy_w        = 1.0         # w_e: energy penalty scale (w_e)
    pbrs_E_ref_v         = 0.01        # kWh/m²/step energy normalisation constant
    reward_type_v        = 'pbrs'

    # Known env defaults — saved verbatim to config.json so the visualiser can
    # recreate the exact environment (avoids introspecting VecEnv subprocesses).
    _STEP_PERIOD        = 1800
    _PREDICTIVE_STEP    = 2 * _STEP_PERIOD   # 3600 s → 60-min forecast interval
    _PREDICTIVE_PERIOD  = 4 * _PREDICTIVE_STEP   # 14400 s → 4 steps: t+0, t+60, t+120, t+180 min
    _MAX_EPISODE_LENGTH = 3 * 24 * 3600
    _WARMUP_PERIOD      = 24 * 3600
    _OBSERVATIONS       = {
        'reaTZon_y': (280.,  310.),
        'TDryBul':   (265,   303),
        'HDirNor':   (0,     862),
        # 'time' intentionally excluded — handled by SinCosTimeWrapper (2 dims)
    }
    _SCENARIO = {'electricity_price': 'constant'}
    _COMFORT_SCHEDULE = CustomRewardWrapper.DEFAULT_COMFORT_SCHEDULE
    _COMFORT_RANGE    = (294.15, 297.15)   # day default

    # ─────────────────────────────────────────────────────────────────────────
    env      = None
    eval_env = None

    print("=" * 80)
    print("BOPTEST RL Training")
    print("=" * 80)

    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name  = f"{algorithm}_PBRS_cw{int(pbrs_comfort_w)}_ew{int(pbrs_energy_w)}_{N_ENVS}envs_{timestamp}"
        save_dir  = os.path.join("trained_models", run_name)
        os.makedirs(save_dir, exist_ok=True)

        print(f"Run name:       {run_name}")
        print(f"Save directory: {save_dir}")
        print(f"Parallel envs:  {N_ENVS}")
        print(f"\nNote: BOPTEST server must be started with at least {N_ENVS} workers:")
        print(f"  docker compose up web worker provision --scale worker={N_ENVS}")

        # ── Create vectorized training environment ────────────────────────────
        print(f"\nCreating {N_ENVS} parallel BOPTEST environments...")
        env = make_boptest_vec_env(
            n_envs=N_ENVS,
            url=URL,
            use_subprocess=True,    # each env in its own process → true parallelism
            testcase=TESTCASE,
            use_custom_reward=True,
            reward_type=reward_type_v,
            pbrs_gamma=pbrs_gamma_v,
            pbrs_comfort_weight=pbrs_comfort_w,
            pbrs_energy_weight=pbrs_energy_w,
            pbrs_E_ref=pbrs_E_ref_v,
            use_discrete_actions=False,
            predictive_period=_PREDICTIVE_PERIOD,
            predictive_step=_PREDICTIVE_STEP,
            use_time_observation=True,
            use_zone_temp_delta=True,
            random_start_time=True,   # expose agent to all seasons/times for generalisation
        )
        print(f"Observation space: {env.observation_space.shape}")
        print(f"Action space:      {env.action_space}")

        # ── Train ─────────────────────────────────────────────────────────────
        print("\n" + "=" * 80)
        print(f"Training SAC Agent  ({N_ENVS} envs × steps = total experience)")
        print("=" * 80)

        total_timesteps = 80000   # 10,000 steps per env × 8 envs

        print(f"\nVectorised steps:        {total_timesteps:,}")
        print(f"Effective env steps:     {total_timesteps * N_ENVS:,}  ({N_ENVS} envs)")

        model, kpi_history = train_sac(
            env,
            total_timesteps=total_timesteps,
            learning_rate=3e-4,
            gamma=_SAC_GAMMA,   # must equal pbrs_gamma_v for policy-invariance guarantee
            verbose=1,
            save_dir=save_dir,
            checkpoint_freq=50000,
        )

        # ── Save model + graph ────────────────────────────────────────────────
        model_path = os.path.join(save_dir, "final_model")
        model.save(model_path)
        print(f"\nModel saved to {model_path}.zip")

        try:
            graph_path = os.path.join(save_dir, "training_progress.png")
            _plot_training_progress(kpi_history, graph_path)
            print(f"Training progress graph saved to {graph_path}")
        except Exception as e:
            print(f"Warning: could not save training progress graph: {e}")

        # ── Save config ───────────────────────────────────────────────────────
        # We build config from the known local variables rather than
        # introspecting VecEnv subprocesses (which requires env_method calls).
        config = {
            'algorithm':         algorithm,
            'testcase':          TESTCASE,
            'n_envs':            N_ENVS,
            'reward_type':       reward_type_v,
            'pbrs_gamma':        pbrs_gamma_v,
            'pbrs_comfort_weight': pbrs_comfort_w,
            'pbrs_energy_weight':  pbrs_energy_w,
            'pbrs_E_ref':          pbrs_E_ref_v,
            'sac_gamma':         _SAC_GAMMA,
            'total_timesteps':          total_timesteps,
            'effective_env_steps':      total_timesteps * N_ENVS,
            'learning_rate':            3e-4,
            'timestamp':                timestamp,
            'observation_dim':          env.observation_space.shape[0],
            'action_space':             str(env.action_space),
            'env_params': {
                'predictive_period':  _PREDICTIVE_PERIOD,
                'predictive_step':    _PREDICTIVE_STEP,
                'regressive_period':  0,
                'max_episode_length': _MAX_EPISODE_LENGTH,
                'step_period':        _STEP_PERIOD,
                'warmup_period':      _WARMUP_PERIOD,
                'random_start_time':  True,
                'start_time':         0,
                'comfort_temp_range': list(_COMFORT_RANGE),
                'use_discrete_actions': False,
                'scenario':           _SCENARIO,
                'observations':       {k: list(v) for k, v in _OBSERVATIONS.items()},
                'use_time_observation': True,
                'comfort_schedule':   [[e[0], e[1], list(e[2])] for e in _COMFORT_SCHEDULE],
                'reward_type':        reward_type_v,
                'pbrs_gamma':         pbrs_gamma_v,
                'pbrs_comfort_weight': pbrs_comfort_w,
                'pbrs_energy_weight':  pbrs_energy_w,
                'pbrs_E_ref':          pbrs_E_ref_v,
                'use_zone_temp_delta': True,
            }
        }

        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        with open(os.path.join(save_dir, 'kpi_history.pkl'), 'wb') as f:
            pickle.dump(kpi_history, f)

        # ── Close training VecEnv before evaluation ───────────────────────────
        # The VecEnv holds N_ENVS BOPTEST workers. Closing it now frees them
        # back to the server pool so the eval env can acquire one.
        # Setting env=None prevents the finally block from closing it a second time.
        print("\nClosing training environments to free BOPTEST workers...")
        env.close()
        env = None
        print(f"✓ {N_ENVS} training workers released")

        # ── Evaluate ──────────────────────────────────────────────────────────
        # Evaluation uses a fresh single env (not the VecEnv) so episodes run
        # sequentially and can be tracked per-episode.
        print("\n" + "=" * 80)
        print("Evaluating Agent  (single env)")
        print("=" * 80)

        eval_env = create_env(
            url=URL,
            testcase=TESTCASE,
            use_custom_reward=True,
            reward_type=reward_type_v,
            pbrs_gamma=pbrs_gamma_v,
            pbrs_comfort_weight=pbrs_comfort_w,
            pbrs_energy_weight=pbrs_energy_w,
            pbrs_E_ref=pbrs_E_ref_v,
            use_discrete_actions=False,
            predictive_period=_PREDICTIVE_PERIOD,
            predictive_step=_PREDICTIVE_STEP,
            use_time_observation=True,
            use_zone_temp_delta=True,
            _register_global=False,
        )

        episode_rewards, episode_kpis = evaluate_agent(eval_env, model, n_episodes=5)

        eval_results = {
            'episode_rewards': episode_rewards,
            'episode_kpis':    episode_kpis,
            'summary': {
                'avg_reward':    float(np.mean(episode_rewards)),
                'std_reward':    float(np.std(episode_rewards)),
                'avg_energy':    float(np.mean([k.get('ener_tot', 0) for k in episode_kpis])),
                'avg_discomfort':float(np.mean([k.get('tdis_tot', 0) for k in episode_kpis])),
                'avg_cost':      float(np.mean([k.get('cost_tot', 0) for k in episode_kpis])),
                'avg_emissions': float(np.mean([k.get('emis_tot', 0) for k in episode_kpis])),
            }
        }

        with open(os.path.join(save_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(eval_results, f, indent=2)

        print("\n" + "=" * 80)
        print("Training Summary")
        print("=" * 80)
        print(f"Average reward:     {eval_results['summary']['avg_reward']:.3f} ± {eval_results['summary']['std_reward']:.3f}")
        print(f"Average energy:     {eval_results['summary']['avg_energy']:.4f}")
        print(f"Average discomfort: {eval_results['summary']['avg_discomfort']:.4f}")
        print(f"Average cost:       {eval_results['summary']['avg_cost']:.4f}")
        print(f"Average emissions:  {eval_results['summary']['avg_emissions']:.4f}")
        print(f"\nAll results saved to: {save_dir}")
        print("\nTraining complete!")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        traceback.print_exc()
    finally:
        print("\nCleaning up...")
        for closing_env, label in [(eval_env, "eval env"), (env, "training envs")]:
            if closing_env is not None:
                try:
                    closing_env.close()
                    print(f"✓ {label} closed")
                except Exception as e:
                    print(f"⚠ Cleanup error ({label}): {e}")

        print("\nIf BOPTEST is stuck, run: python cleanup_boptest.py")
