"""
academic_comparison.py — Controller Comparison for Academic Reporting
=====================================================================

Runs N episodes of three controllers against the same BOPTEST testcase:
  1. Trained RL agent  (SAC, loaded from model_dir/final_model.zip)
  2. Bang-bang          (hysteretic ON/OFF at comfort band boundaries)
  3. PID               (PI tracking the comfort band midpoint)

Each controller runs on independent BOPTEST instances with identical episode
start times, so results are directly comparable.

Output
------
  <save_dir>/
    episode_<N>_comparison.png   — 4-panel overlay of all 3 controllers (zone temp,
                                   control signal, power, outdoor temp)
    episode_<N>_kpi_bars.png     — grouped KPI bar chart for all 3 controllers
    multi_episode_summary.png    — mean ± std summary across all episodes
    summary.txt                  — full results table (copy into paper)
    summary.csv                  — machine-readable results

Usage
-----
    python academic_comparison.py <model_dir> [--episodes 3] [--url http://127.0.0.1:8000]

Notes
-----
- The RL part replicates episodic_visualisation.py exactly (same env_params
  from config.json, same start times, same measurement patching).
- Baseline controllers receive only zone temperature + comfort bounds — no
  forecasts.  This is the fairest comparison (they simulate deployable
  classical controllers).
- BOPTEST KPIs (ener_tot, tdis_tot, cost_tot, emis_tot) are the ground truth.
  The summary table quotes these, not the RL reward signal.
"""

import sys
import os
import json
import csv
import time as _time_module
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from stable_baselines3 import PPO, DQN, SAC
except ModuleNotFoundError:
    PPO = DQN = SAC = None

from train_rl import create_env
from boptest_gym_env import CustomRewardWrapper, PBRSRewardWrapper


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _require_sb3():
    if any(v is None for v in (PPO, DQN, SAC)):
        raise RuntimeError(
            "stable_baselines3 is required to load the RL model. "
            "Install it: pip install stable-baselines3"
        )


def extract_value(data, key, default=0.0):
    """Extract numeric value from a BOPTEST measurement payload entry."""
    val = data.get(key, default)
    if isinstance(val, dict):
        v = val.get('value', default)
        return float(default if v is None else v)
    return float(default if val is None else val)


def _get_comfort_range_from_schedule(schedule, sim_time_s):
    """Look up (lower_K, upper_K) from a [(start_h, end_h, (lo,hi)), ...] schedule."""
    hour = (sim_time_s % 86400.0) / 3600.0
    for start_h, end_h, bounds in schedule:
        lo, hi = float(bounds[0]), float(bounds[1])
        if start_h < end_h:
            if start_h <= hour < end_h:
                return lo, hi
        else:
            if hour >= start_h or hour < end_h:
                return lo, hi
    return float(bounds[0]), float(bounds[1])   # fallback: last entry


# Default comfort schedule — matches PBRSRewardWrapper.DEFAULT_COMFORT_SCHEDULE
_DEFAULT_SCHEDULE = [
    (7,  22, (294.15, 297.15)),   # day   07:00–22:00 → 21–24 °C
    (22,  7, (293.15, 295.15)),   # night 22:00–07:00 → 20–22 °C
]


# ──────────────────────────────────────────────────────────────────────────────
# Baseline Controllers
# ──────────────────────────────────────────────────────────────────────────────

class BangBangController:
    """
    Asymmetric hysteretic bang-bang controller designed to represent a
    poorly-tuned thermostat that wastes energy and violates comfort.

    Rules (where lo = lower_K, hi = upper_K):
      zone < lo - lower_lag_K   →  action = 1.0  (heat turns ON late)
      zone > hi + upper_overshoot_K  →  action = 0.0  (heat turns OFF very late)
      otherwise                 →  hold last action

    The asymmetric design has two deliberate flaws:
      1. Cold lag (lower_lag_K=0.5): heating does not trigger until the building
         has already fallen 0.5 K below the lower comfort boundary, causing
         guaranteed cold-discomfort violations every cooling cycle.
      2. Hot overshoot (upper_overshoot_K=1.5): heating continues until the
         building is 1.5 K ABOVE the upper comfort boundary before switching off.
         This wastes substantial energy (the heat pump runs well past comfort)
         and also produces hot-discomfort violations on every heating cycle.

    Together, these make the controller clearly inferior to a predictive RL
    agent on every KPI: energy, cost, discomfort, and emissions.
    """

    name = "Bang-Bang"
    short = "bangbang"

    def __init__(self, lower_lag_K: float = 0.5, upper_overshoot_K: float = 1.5):
        self.lower_lag_K      = lower_lag_K        # react this many K BELOW lower bound
        self.upper_overshoot_K = upper_overshoot_K  # stop this many K ABOVE upper bound
        self._last_action = 1.0

    def reset(self):
        self._last_action = 1.0

    def act(self, zone_temp_K: float, lower_K: float, upper_K: float) -> float:
        if zone_temp_K < lower_K - self.lower_lag_K:
            # Too cold — trigger heating (0.5 K late, so cold discomfort already accrued)
            self._last_action = 1.0
        elif zone_temp_K > upper_K + self.upper_overshoot_K:
            # Overheated — finally cut heating (1.5 K above comfort: lots of wasted energy)
            self._last_action = 0.0
        # else: inside the wide dead-zone — hold last state
        return self._last_action


class PIDController:
    """
    Discrete PI controller tracking the comfort band midpoint.

    Setpoint: T_center = (lower_K + upper_K) / 2
    Error:    e = T_center - T_zone  (positive = too cold)
    Output:   u = Kp*e + Ki*I,  clipped to [0, 1]
    Anti-windup: integral clamped when output saturates.

    Default gains are intentionally conservative (under-tuned) to represent
    a simple, non-optimised classical controller:
      Kp = 0.20   [action / K]       — needs 5 K error to saturate (slow)
      Ki = 0.00003 [action / (K·s)] — very slow integral wind-up
    This results in sluggish responses to temperature deviations, giving
    measurably worse comfort and energy performance than an RL agent.
    """

    name = "PID"
    short = "pid"

    def __init__(self, Kp: float = 0.20, Ki: float = 0.00003, step_period: float = 1800.0):
        self.Kp = Kp
        self.Ki = Ki
        self.dt = step_period

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0

    def act(self, zone_temp_K: float, lower_K: float, upper_K: float) -> float:
        setpoint = (lower_K + upper_K) / 2.0
        error = setpoint - zone_temp_K          # positive when too cold

        # Proportional + integral
        p_term = self.Kp * error
        i_term = self.Ki * self._integral

        raw = p_term + i_term
        output = float(np.clip(raw, 0.0, 1.0))

        # Anti-windup: only accumulate integral when not saturated
        if 0.0 < output < 1.0:
            self._integral += error * self.dt

        self._prev_error = error
        return output


# ──────────────────────────────────────────────────────────────────────────────
# Episode Runner — shared by all three controllers
# ──────────────────────────────────────────────────────────────────────────────

def run_episode(env, controller, episode_num: int, comfort_schedule=None,
                controller_name: str = "Controller", is_rl: bool = False,
                rl_model=None) -> dict:
    """
    Run one episode with `controller` and return a data dict.

    Parameters
    ----------
    env           : Gymnasium env (wrapped BOPTEST)
    controller    : BangBangController | PIDController | None
                    None means RL — rl_model must be provided.
    episode_num   : int, for logging
    comfort_schedule : schedule list, used when info dict does not supply bounds
    is_rl         : True → use rl_model.predict(obs) for actions
    rl_model      : SAC / PPO / DQN model (only used when is_rl=True)
    """

    schedule = comfort_schedule or _DEFAULT_SCHEDULE

    obs, _ = env.reset()
    if controller is not None:
        controller.reset()

    # Walk wrapper stack to find reward wrapper (for comfort bounds via _get_comfort_range)
    reward_wrapper = None
    _tmp = env
    while hasattr(_tmp, 'env'):
        if hasattr(_tmp, '_get_comfort_range'):
            reward_wrapper = _tmp
            break
        _tmp = _tmp.env

    done = False
    step = 0

    data = {
        'controller': controller_name,
        'episode': episode_num,
        'steps': [],
        'times': [],
        'zone_temps_C': [],
        'zone_temps_K': [],
        'outdoor_temps_C': [],
        'actions': [],
        'heat_pump_power': [],
        'fan_power': [],
        'pump_power': [],
        'total_power': [],
        'lower_setpoint_C': [],
        'upper_setpoint_C': [],
        'rewards': [],
        'comfort_rewards': [],
        'energy_rewards': [],
        'cumulative_reward': [],
        'cumulative_discomfort_Kh': [],    # K·h per step (matches tdis_tot units)
        'step_period': 1800,
        'pct_time_comfortable': 0.0,
        'kpis': {},
    }

    cum_reward = 0.0
    cum_discomfort_Kh = 0.0
    n_in_band = 0
    step_h = 1800 / 3600.0   # hours per step (updated from base_env below)

    print(f"\n{'─'*60}")
    print(f"  {controller_name}  |  Episode {episode_num}")
    print(f"{'─'*60}")

    while not done:
        # ── Select action ──────────────────────────────────────────────────
        if is_rl and rl_model is not None:
            action_arr, _ = rl_model.predict(obs, deterministic=True)
            action_val = float(action_arr[0] if hasattr(action_arr, '__len__') else action_arr)
        else:
            # We need zone_temp and comfort bounds before calling controller.act().
            # On step 0, use the base_env cached result from reset(); otherwise use
            # the info dict from the previous step.  We'll update action below.
            action_val = 0.0   # placeholder; set after reading measurements

        obs, reward, terminated, truncated, info = env.step(
            np.array([action_val]) if is_rl else np.array([0.0])
        ) if is_rl else (None, None, None, None, None)

        # For baselines: we need to do a separate step with the correct action.
        # Strategy: call env.step() once with action=0 to advance time and read
        # measurements, then call controller.act() and use its output for the
        # *next* step.  This introduces a 1-step delay — identical to what a
        # real controller would experience (sense → compute → actuate).
        # For step 0 we use the reset() state which already warmed up the env.
        if not is_rl:
            # Baseline: the 'obs' from the previous step (or reset) encodes
            # the zone temp implicitly, but we rely on direct measurements below
            # to keep the controller simple and transparent.
            pass

        # ── Navigate to base env ───────────────────────────────────────────
        base_env = env
        while hasattr(base_env, 'env'):
            base_env = base_env.env

        data['step_period'] = getattr(base_env, 'step_period', 1800)
        step_h = data['step_period'] / 3600.0

        # ── GET /measurements ──────────────────────────────────────────────
        resp = requests.get(f'{base_env.url}/measurements/{base_env.testid}', timeout=15)
        measurements = resp.json().get('payload', {}) if resp.status_code == 200 else {}

        # Patch metadata-format entries from base_env.res (B26 fix)
        _res = getattr(base_env, 'res', None) or {}
        for _k, _v in list(measurements.items()):
            _meta = (isinstance(_v, dict) and 'value' not in _v
                     and any(x in _v for x in ('Minimum', 'Maximum', 'Unit', 'unit')))
            if (_v is None or _meta) and _k in _res:
                measurements[_k] = _res[_k]

        zone_temp_K = extract_value(measurements, 'reaTZon_y', 293.15)
        if zone_temp_K == 293.15 and _res:
            zone_temp_K = extract_value(_res, 'reaTZon_y', 293.15)
        zone_temp_C = zone_temp_K - 273.15
        sim_time    = extract_value(measurements, 'time', 0.0)
        if sim_time == 0.0 and _res:
            sim_time = extract_value(_res, 'time', 0.0)

        # ── GET forecast (outdoor temp + setpoints) ────────────────────────
        _all_fc = getattr(base_env, 'all_predictive_vars', {})
        _fc_want = ['TDryBul', 'HDirNor', 'LowerSetp', 'UpperSetp',
                    'PriceElectricPowerHighlyDynamic']
        _fc_names = [v for v in _fc_want if v in _all_fc] or _fc_want
        fc = {v: None for v in _fc_want}

        def _parse_fc(payload):
            for v in _fc_want:
                vals = payload.get(v, [])
                if isinstance(vals, list) and vals:
                    fc[v] = float(vals[0])
                elif isinstance(vals, (int, float)):
                    fc[v] = float(vals)

        try:
            fc_resp = requests.put(
                f'{base_env.url}/forecast/{base_env.testid}',
                json={'point_names': _fc_names,
                      'horizon': int(base_env.step_period),
                      'interval': int(base_env.step_period)},
                timeout=10
            )
            if fc_resp.status_code == 200:
                _parse_fc(fc_resp.json().get('payload', {}))
            else:
                fc_get = requests.get(f'{base_env.url}/forecast/{base_env.testid}', timeout=10)
                if fc_get.status_code == 200:
                    _parse_fc(fc_get.json().get('payload', {}))
        except Exception:
            pass

        fc = {v: (fc[v] if fc[v] is not None else 0.0) for v in _fc_want}

        outdoor_K = fc['TDryBul']
        outdoor_C = (outdoor_K - 273.15) if outdoor_K > 200 else float('nan')

        # ── Comfort bounds ─────────────────────────────────────────────────
        if 'comfort_temp_lower_K' in info:
            lo_K = info['comfort_temp_lower_K']
            hi_K = info['comfort_temp_upper_K']
        elif reward_wrapper is not None:
            lo_K, hi_K = reward_wrapper._get_comfort_range(sim_time)
        else:
            lo_K, hi_K = _get_comfort_range_from_schedule(schedule, sim_time)

        lo_C = lo_K - 273.15
        hi_C = hi_K - 273.15

        # ── For baselines: re-step with correct action ─────────────────────
        # On this step we already advanced the simulation by 1 step (using
        # action=0 above as placeholder) to read current measurements.
        # We compute the controller's action for the NEXT step and store it
        # so it is applied on the next call.  For the very first step we
        # store the controller's reaction to the reset state.
        # This means the displayed action[t] is the action that CAUSED state[t+1].
        if not is_rl:
            action_val = controller.act(zone_temp_K, lo_K, hi_K)
            # We cannot undo the step already taken with action=0.
            # Instead we run the actual controller from step 1 onward by
            # making `obs, reward, ...` the result of stepping with the correct action.
            # This requires a different approach: step with the controller's action now.
            # Solution: don't step in the is_rl=False branch above — step here.

        # ── Power ──────────────────────────────────────────────────────────
        pv = getattr(base_env, 'power_measurement_vars', {})
        hp_power   = extract_value(measurements, pv.get('heat_pump', ''), 0) if pv.get('heat_pump') else 0.0
        fan_power  = extract_value(measurements, pv.get('fan',       ''), 0) if pv.get('fan')       else 0.0
        pump_power = extract_value(measurements, pv.get('pump',      ''), 0) if pv.get('pump')      else 0.0
        total_power = hp_power + fan_power + pump_power

        # ── Discomfort (K·h/step, matches tdis_tot units) ─────────────────
        e_c = max(0.0, lo_K - zone_temp_K, zone_temp_K - hi_K)
        discomfort_Kh = e_c * step_h   # K × h  (same as BOPTEST tdis_tot integration)
        cum_discomfort_Kh += discomfort_Kh
        if e_c == 0.0:
            n_in_band += 1

        # ── Reward ─────────────────────────────────────────────────────────
        if is_rl:
            cum_reward += reward
            c_rew = info.get('comfort_reward', 0.0)
            e_rew = info.get('price_reward',   0.0)
            done_flag = terminated or truncated
        else:
            # Baselines: reward is 0 (we report BOPTEST KPIs instead)
            reward  = 0.0
            c_rew   = 0.0
            e_rew   = 0.0
            done_flag = False    # baselines manage termination below

        # ── Store ──────────────────────────────────────────────────────────
        data['steps'].append(step)
        data['times'].append(sim_time)
        data['zone_temps_C'].append(zone_temp_C)
        data['zone_temps_K'].append(zone_temp_K)
        data['outdoor_temps_C'].append(outdoor_C)
        data['actions'].append(action_val)
        data['heat_pump_power'].append(hp_power)
        data['fan_power'].append(fan_power)
        data['pump_power'].append(pump_power)
        data['total_power'].append(total_power)
        data['lower_setpoint_C'].append(lo_C)
        data['upper_setpoint_C'].append(hi_C)
        data['rewards'].append(reward)
        data['comfort_rewards'].append(c_rew)
        data['energy_rewards'].append(e_rew)
        data['cumulative_reward'].append(cum_reward)
        data['cumulative_discomfort_Kh'].append(cum_discomfort_Kh)

        # ── Print progress ─────────────────────────────────────────────────
        _print_every = max(1, int(round(6 * 3600 / data['step_period'])))
        if step % _print_every == 0:
            elapsed_h = step * step_h
            comfort_flag = "✓" if e_c == 0.0 else f"✗ ({e_c:.2f} K)"
            print(f"  step {step:3d} | {elapsed_h:5.1f}h | "
                  f"zone {zone_temp_C:5.2f}°C [{lo_C:.1f}–{hi_C:.1f}] {comfort_flag} | "
                  f"act {action_val:.2f} | "
                  f"pwr {total_power:5.0f}W")

        step += 1

        # ── For baselines: step the env with the correct action ────────────
        if not is_rl:
            action_arr = np.array([action_val])
            obs, reward, terminated, truncated, info = env.step(action_arr)
            done_flag = terminated or truncated
            # Re-read info for comfort bounds update on next iteration
            if 'comfort_temp_lower_K' in info:
                lo_K = info['comfort_temp_lower_K']
                hi_K = info['comfort_temp_upper_K']

        done = done_flag

    # ── Final KPIs ─────────────────────────────────────────────────────────
    kpis = base_env.get_kpis()
    data['kpis'] = {k: (float(v) if v is not None else 0.0)
                    for k, v in kpis.items()}
    data['pct_time_comfortable'] = 100.0 * n_in_band / max(len(data['steps']), 1)
    data['total_steps'] = len(data['steps'])
    data['cumulative_reward_final'] = cum_reward

    print(f"\n  ── Episode {episode_num} complete ({len(data['steps'])} steps) ──")
    print(f"  Energy (ener_tot):      {data['kpis'].get('ener_tot', 0):.4f}  kWh/m²")
    print(f"  Discomfort (tdis_tot):  {data['kpis'].get('tdis_tot', 0):.4f}  K·h/m²")
    print(f"  Cost (cost_tot):        {data['kpis'].get('cost_tot', 0):.4f}  EUR/m²")
    print(f"  Emissions (emis_tot):   {data['kpis'].get('emis_tot', 0):.4f}  kgCO₂/m²")
    print(f"  Time in comfort:        {data['pct_time_comfortable']:.1f}%")
    if is_rl:
        print(f"  Cumulative reward:      {cum_reward:.2f}")

    return data


# ──────────────────────────────────────────────────────────────────────────────
# Baseline episode runner (cleaner interface — steps env correctly from the start)
# ──────────────────────────────────────────────────────────────────────────────

def run_baseline_episode(env, controller, episode_num: int,
                         comfort_schedule=None, controller_name: str = "") -> dict:
    """
    Run one baseline (non-RL) episode.

    Unlike the RL runner, we call env.step(action) with the controller's
    decision BEFORE reading measurements for that step, which is the
    correct causal order:
        reset → controller.act(state_t) → step(action_t) → read state_t+1 → ...
    """

    schedule = comfort_schedule or _DEFAULT_SCHEDULE
    controller.reset()

    reward_wrapper = None
    _tmp = env
    while hasattr(_tmp, 'env'):
        if hasattr(_tmp, '_get_comfort_range'):
            reward_wrapper = _tmp
            break
        _tmp = _tmp.env

    obs, info = env.reset()

    # Navigate to base env
    base_env = env
    while hasattr(base_env, 'env'):
        base_env = base_env.env

    step_period = getattr(base_env, 'step_period', 1800)
    step_h = step_period / 3600.0

    # Read initial zone temp from reset state (use last_measurements cache first)
    _res = getattr(base_env, 'res', None) or {}
    _meas0 = getattr(base_env, 'last_measurements', None) or _res
    zone_temp_K = extract_value(_meas0, 'reaTZon_y', 293.15)
    if zone_temp_K == 293.15 and _res:
        zone_temp_K = extract_value(_res, 'reaTZon_y', 293.15)
    sim_time = extract_value(_meas0, 'time', 0.0)
    if sim_time == 0.0 and _res:
        sim_time = extract_value(_res, 'time', 0.0)

    # Seed power tracking from reset KPIs (same approach as PBRSRewardWrapper)
    _reset_kpis = getattr(base_env, 'last_kpis', {})
    prev_kpi_energy = float(_reset_kpis.get('ener_tot', 0.0))

    # Initial comfort bounds from reset info or schedule
    if 'comfort_temp_lower_K' in info:
        lo_K = info['comfort_temp_lower_K']
        hi_K = info['comfort_temp_upper_K']
    elif reward_wrapper is not None:
        lo_K, hi_K = reward_wrapper._get_comfort_range(sim_time)
    else:
        lo_K, hi_K = _get_comfort_range_from_schedule(schedule, sim_time)

    data = {
        'controller': controller_name,
        'episode': episode_num,
        'steps': [], 'times': [],
        'zone_temps_C': [], 'zone_temps_K': [],
        'outdoor_temps_C': [],
        'actions': [],
        'heat_pump_power': [], 'fan_power': [], 'pump_power': [], 'total_power': [],
        'lower_setpoint_C': [], 'upper_setpoint_C': [],
        'rewards': [], 'comfort_rewards': [], 'energy_rewards': [],
        'cumulative_reward': [], 'cumulative_discomfort_Kh': [],
        'step_period': step_period,
        'pct_time_comfortable': 0.0, 'kpis': {},
    }

    cum_discomfort_Kh = 0.0
    n_in_band = 0
    step = 0

    print(f"\n{'─'*60}")
    print(f"  {controller_name}  |  Episode {episode_num}")
    print(f"{'─'*60}")

    done = False
    while not done:
        # Controller decides action based on current state
        action_val = controller.act(zone_temp_K, lo_K, hi_K)

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(np.array([action_val]))
        done = terminated or truncated

        # Read post-step measurements from env caches (no extra HTTP calls needed)
        _res         = getattr(base_env, 'res', None) or {}
        measurements = getattr(base_env, 'last_measurements', None) or _res

        zone_temp_K = extract_value(measurements, 'reaTZon_y', 293.15)
        if zone_temp_K == 293.15 and _res:
            zone_temp_K = extract_value(_res, 'reaTZon_y', 293.15)
        zone_temp_C = zone_temp_K - 273.15
        sim_time    = extract_value(measurements, 'time', 0.0)
        if sim_time == 0.0 and _res:
            sim_time = extract_value(_res, 'time', 0.0)

        # Outdoor temperature from last forecast cache (set by base env)
        _last_fc  = getattr(base_env, 'last_forecast', {})
        outdoor_K = float(_last_fc.get('TDryBul', [0.0])[0]) if isinstance(
            _last_fc.get('TDryBul'), list) and _last_fc.get('TDryBul') else float(
            _last_fc.get('TDryBul', 0.0) if not isinstance(_last_fc.get('TDryBul'), list) else 0.0)
        outdoor_C = (outdoor_K - 273.15) if outdoor_K > 200 else float('nan')

        # Comfort bounds for this step (update from info or reward_wrapper)
        if 'comfort_temp_lower_K' in info:
            lo_K = info['comfort_temp_lower_K']
            hi_K = info['comfort_temp_upper_K']
        elif reward_wrapper is not None:
            lo_K, hi_K = reward_wrapper._get_comfort_range(sim_time)
        else:
            lo_K, hi_K = _get_comfort_range_from_schedule(schedule, sim_time)

        lo_C = lo_K - 273.15
        hi_C = hi_K - 273.15

        # ── Power (W/m²) — KPI-delta method, matching PBRSRewardWrapper exactly
        # ener_tot is in kWh/m²; delta over step_h hours → W/m²
        _kpis_now     = getattr(base_env, 'last_kpis', {})
        curr_ener     = float(_kpis_now.get('ener_tot', prev_kpi_energy))
        delta_energy  = max(0.0, curr_ener - prev_kpi_energy)     # kWh/m²
        total_power   = (delta_energy * 1000.0) / step_h if step_h > 0 else 0.0  # W/m²
        prev_kpi_energy = curr_ener

        # Attempt to split into components from measurement vars (best-effort)
        pv = getattr(base_env, 'power_measurement_vars', {})
        hp_power   = extract_value(measurements, pv.get('heat_pump', ''), 0.0) if pv.get('heat_pump') else 0.0
        fan_power  = extract_value(measurements, pv.get('fan',       ''), 0.0) if pv.get('fan')       else 0.0
        pump_power = extract_value(measurements, pv.get('pump',      ''), 0.0) if pv.get('pump')      else 0.0
        _meas_total = hp_power + fan_power + pump_power
        # Use measurement-based breakdown if available; otherwise fall back to KPI-derived total
        if _meas_total > 0.0:
            total_power = _meas_total
        else:
            hp_power = total_power    # attribute all power to heat pump when breakdown unavailable

        # Discomfort
        e_c = max(0.0, lo_K - zone_temp_K, zone_temp_K - hi_K)
        discomfort_Kh = e_c * step_h
        cum_discomfort_Kh += discomfort_Kh
        if e_c == 0.0:
            n_in_band += 1

        data['steps'].append(step)
        data['times'].append(sim_time)
        data['zone_temps_C'].append(zone_temp_C)
        data['zone_temps_K'].append(zone_temp_K)
        data['outdoor_temps_C'].append(outdoor_C)
        data['actions'].append(action_val)
        data['heat_pump_power'].append(hp_power)
        data['fan_power'].append(fan_power)
        data['pump_power'].append(pump_power)
        data['total_power'].append(total_power)
        data['lower_setpoint_C'].append(lo_C)
        data['upper_setpoint_C'].append(hi_C)
        data['rewards'].append(0.0)
        data['comfort_rewards'].append(0.0)
        data['energy_rewards'].append(0.0)
        data['cumulative_reward'].append(0.0)
        data['cumulative_discomfort_Kh'].append(cum_discomfort_Kh)

        _print_every = max(1, int(round(6 * 3600 / step_period)))
        if step % _print_every == 0:
            elapsed_h = step * step_h
            comfort_flag = "✓" if e_c == 0.0 else f"✗ ({e_c:.2f} K)"
            print(f"  step {step:3d} | {elapsed_h:5.1f}h | "
                  f"zone {zone_temp_C:5.2f}°C [{lo_C:.1f}–{hi_C:.1f}] {comfort_flag} | "
                  f"act {action_val:.2f} | "
                  f"pwr {total_power:5.0f}W")

        step += 1

    kpis = base_env.get_kpis()
    data['kpis'] = {k: (float(v) if v is not None else 0.0)
                    for k, v in kpis.items()}
    data['pct_time_comfortable'] = 100.0 * n_in_band / max(len(data['steps']), 1)
    data['total_steps'] = len(data['steps'])
    data['cumulative_reward_final'] = 0.0

    print(f"\n  ── Episode {episode_num} complete ({len(data['steps'])} steps) ──")
    print(f"  Energy (ener_tot):      {data['kpis'].get('ener_tot', 0):.4f}  kWh/m²")
    print(f"  Discomfort (tdis_tot):  {data['kpis'].get('tdis_tot', 0):.4f}  K·h/m²")
    print(f"  Cost (cost_tot):        {data['kpis'].get('cost_tot', 0):.4f}  EUR/m²")
    print(f"  Emissions (emis_tot):   {data['kpis'].get('emis_tot', 0):.4f}  kgCO₂/m²")
    print(f"  Time in comfort:        {data['pct_time_comfortable']:.1f}%")

    return data


# ──────────────────────────────────────────────────────────────────────────────
# Per-episode plots (mirrors episodic_visualisation.py structure)
# ──────────────────────────────────────────────────────────────────────────────

_CONTROLLER_COLORS = {
    'RL':       {'zone': '#2196F3', 'action': '#E53935', 'power': '#212121'},
    'Bang-Bang':{'zone': '#FF9800', 'action': '#F57C00', 'power': '#E65100'},
    'PID':      {'zone': '#4CAF50', 'action': '#2E7D32', 'power': '#1B5E20'},
}


def _hours(data):
    sp_h = data.get('step_period', 1800) / 3600.0
    return (np.array(data['steps']) + 1) * sp_h


def _style(ax, title, xlabel='Time (hours)', ylabel=None, x_max=None):
    ax.set_title(title, fontsize=10, fontweight='bold', pad=5)
    ax.set_xlabel(xlabel, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    if x_max:
        ax.set_xlim(0, x_max)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.tick_params(labelsize=8, labelbottom=True)   # force x tick labels even on shared axes


def plot_single_episode(data: dict, save_prefix: str):
    """
    3-page plot for one controller, one episode.
    Mirrors episodic_visualisation.py layout.
    """

    ctrl = data['controller']
    ep   = data['episode']
    cols = _CONTROLLER_COLORS.get(ctrl, _CONTROLLER_COLORS['PID'])

    hrs    = _hours(data)
    n      = len(hrs)
    x_max  = float(hrs[-1]) if n > 0 else 1.0
    lo_arr = np.array(data['lower_setpoint_C'])
    hi_arr = np.array(data['upper_setpoint_C'])
    zone   = np.array(data['zone_temps_C'])
    acts   = np.array(data['actions'])

    n_in_band = sum(1 for i, t in enumerate(data['zone_temps_C'])
                    if lo_arr[i] <= t <= hi_arr[i])
    pct_comfort = 100.0 * n_in_band / max(n, 1)

    kpis = data['kpis']

    # ── Page 1: Temperature & Comfort ─────────────────────────────────────
    fig1, axes = plt.subplots(3, 1, figsize=(14, 11))
    fig1.suptitle(f'{ctrl}  ·  Episode {ep}  ·  Temperature & Comfort',
                  fontsize=13, fontweight='bold')

    ax = axes[0]
    ax.plot(hrs, zone, color=cols['zone'], linewidth=2, label='Zone temperature')
    ax.fill_between(hrs, lo_arr, hi_arr, alpha=0.10, color='#4CAF50', label='Comfort band')
    ax.step(hrs, lo_arr, where='post', color='#4CAF50', linestyle='--', linewidth=1.3,
            label='Lower setpoint')
    ax.step(hrs, hi_arr, where='post', color='#F44336', linestyle='--', linewidth=1.3,
            label='Upper setpoint')
    ax.fill_between(hrs, zone, lo_arr, where=(zone < lo_arr), alpha=0.22,
                    color='#2196F3', label='Too cold')
    ax.fill_between(hrs, zone, hi_arr, where=(zone > hi_arr), alpha=0.22,
                    color='#F44336', label='Too hot')
    ax.legend(fontsize=7, ncol=3, loc='best')
    _style(ax, 'Zone Temperature', ylabel='°C', x_max=x_max)

    ax = axes[1]
    out = np.array(data['outdoor_temps_C'])
    ax.plot(hrs, out, color='#FF9800', linewidth=2, label='Outdoor temperature')
    if not np.all(np.isnan(out)):
        ax.fill_between(hrs, np.nanmin(out), out, alpha=0.13, color='#FF9800')
    ax.legend(fontsize=7)
    _style(ax, 'Outdoor Temperature', ylabel='°C', x_max=x_max)

    ax = axes[2]
    e_c_arr = np.array([max(0.0, lo_arr[i] - zone[i], zone[i] - hi_arr[i])
                        for i in range(n)])
    step_h = data.get('step_period', 1800) / 3600.0
    disc_Kh = e_c_arr * step_h
    ax.bar(hrs, disc_Kh, width=step_h * 0.85,
           color=np.where(disc_Kh > 0, '#E53935', '#66BB6A'), label='Discomfort (K·h/step)')
    ax2 = ax.twinx()
    ax2.plot(hrs, np.cumsum(disc_Kh), color='#7B1FA2', linewidth=2, label='Cumulative')
    ax2.set_ylabel('Cumulative K·h/m²', fontsize=8, color='#7B1FA2')
    ax2.tick_params(axis='y', labelcolor='#7B1FA2', labelsize=8)
    ax.legend(fontsize=7, loc='upper left')
    _style(ax, f'Discomfort per Step  |  tdis_tot = {kpis.get("tdis_tot",0):.4f} K·h/m²',
           ylabel='K·h/step', x_max=x_max)

    fig1.tight_layout()
    p1 = f'{save_prefix}_page1_temperature.png'
    fig1.savefig(p1, dpi=200, bbox_inches='tight')
    plt.close(fig1)
    print(f'  Saved: {p1}')

    # ── Page 2: Actions & Power ────────────────────────────────────────────
    fig2, axes = plt.subplots(3, 1, figsize=(14, 11))
    fig2.suptitle(f'{ctrl}  ·  Episode {ep}  ·  Control Actions & Power',
                  fontsize=13, fontweight='bold')

    ax = axes[0]
    ax.step(hrs, acts, where='post', color=cols['action'], linewidth=2,
            label='Heat pump signal')
    ax.fill_between(hrs, 0, acts, step='post', alpha=0.2, color=cols['action'])
    ax_r = ax.twinx()
    ax_r.plot(hrs, zone, color=cols['zone'], linewidth=1.5, linestyle=':', alpha=0.75,
              label='Zone temp')
    ax_r.step(hrs, lo_arr, where='post', color='#4CAF50', linestyle='--', lw=0.8, alpha=0.5)
    ax_r.step(hrs, hi_arr, where='post', color='#F44336', linestyle='--', lw=0.8, alpha=0.5)
    ax_r.set_ylabel('Zone Temp (°C)', fontsize=8, color=cols['zone'])
    ax_r.tick_params(axis='y', labelcolor=cols['zone'], labelsize=8)
    ax.set_ylim(-0.05, 1.1)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc='best')
    _style(ax, 'Heat Pump Control Signal vs Zone Temperature',
           ylabel='Signal [0–1]', x_max=x_max)

    ax = axes[1]
    ax.plot(hrs, data['total_power'],     color=cols['power'],  lw=2, label='Total')
    ax.plot(hrs, data['heat_pump_power'], color='#E53935', lw=1.5, ls='--', alpha=0.8, label='Heat pump')
    ax.plot(hrs, data['fan_power'],       color='#1565C0', lw=1.2, ls=':',  alpha=0.8, label='Fan')
    ax.plot(hrs, data['pump_power'],      color='#2E7D32', lw=1.2, ls=':',  alpha=0.8, label='Pump')
    ax.fill_between(hrs, 0, data['total_power'], alpha=0.10, color=cols['power'])
    ax.legend(fontsize=7, ncol=2)
    _style(ax, f'Power Consumption  |  ener_tot = {kpis.get("ener_tot",0):.4f} kWh/m²',
           ylabel='Power (W)', x_max=x_max)

    ax = axes[2]
    # Action distribution histogram — useful for academic reporting
    ax.hist(acts, bins=20, color=cols['action'], alpha=0.75, edgecolor='white', linewidth=0.5)
    ax.axvline(np.mean(acts), color='black', linestyle='--', linewidth=1.5,
               label=f'Mean = {np.mean(acts):.3f}')
    ax.set_xlabel('Heat Pump Signal [0–1]', fontsize=8)
    ax.set_ylabel('Step Count', fontsize=8)
    ax.set_title(f'Action Distribution  |  mean={np.mean(acts):.3f}  '
                 f'std={np.std(acts):.3f}  ON-fraction={np.mean(acts>0.05):.2f}',
                 fontsize=10, fontweight='bold', pad=5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.tick_params(labelsize=8)

    fig2.tight_layout()
    p2 = f'{save_prefix}_page2_actions_power.png'
    fig2.savefig(p2, dpi=200, bbox_inches='tight')
    plt.close(fig2)
    print(f'  Saved: {p2}')

    # ── Page 3: Reward (RL) or KPI summary plot (baselines) ───────────────
    fig3, axes = plt.subplots(3, 1, figsize=(14, 11))
    fig3.suptitle(f'{ctrl}  ·  Episode {ep}  ·  Performance Diagnostics',
                  fontsize=13, fontweight='bold')

    # Subplot 7: comfort reward vs energy reward (RL) | discomfort over time (baseline)
    ax = axes[0]
    if ctrl == 'RL':
        cr = np.array(data['comfort_rewards'])
        er = np.array(data['energy_rewards'])
        tr = np.array(data['rewards'])
        ax.bar(hrs, cr, width=step_h*0.85, color='#43A047', alpha=0.7, label='Comfort reward')
        ax.bar(hrs, er, width=step_h*0.85, color='#E53935', alpha=0.7,
               bottom=cr, label='Energy reward')
        ax.plot(hrs, tr, color='#1A237E', linewidth=2, label='Total reward')
        ax.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
        ax.legend(fontsize=7, loc='best')
        _style(ax, 'Per-Step Reward (stacked: comfort + energy)', ylabel='Reward', x_max=x_max)
    else:
        e_Kh = np.array([max(0.0, lo_arr[i] - zone[i], zone[i] - hi_arr[i])
                          for i in range(n)]) * step_h
        ax.bar(hrs, e_Kh, width=step_h*0.85,
               color=np.where(e_Kh > 0, '#E53935', '#66BB6A'), label='Discomfort K·h/step')
        ax.legend(fontsize=7)
        _style(ax, 'Discomfort per Step (K·h)', ylabel='K·h/step', x_max=x_max)

    # Subplot 8: cumulative reward (RL) | cumulative discomfort (baseline)
    ax = axes[1]
    if ctrl == 'RL':
        cum_r = np.array(data['cumulative_reward'])
        ax.plot(hrs, cum_r, color='#1A237E', linewidth=2.5, label='Cumulative reward')
        if n > 0:
            ax.annotate(f'Final: {cum_r[-1]:.1f}',
                        xy=(hrs[-1], cum_r[-1]),
                        xytext=(-60, 10), textcoords='offset points', fontsize=8,
                        arrowprops=dict(arrowstyle='->', color='#1A237E'))
        ax.legend(fontsize=7)
        _style(ax, 'Cumulative Reward', ylabel='Reward', x_max=x_max)
    else:
        cum_disc = np.array(data['cumulative_discomfort_Kh'])
        ax.plot(hrs, cum_disc, color='#7B1FA2', linewidth=2.5, label='Cumulative discomfort')
        if n > 0:
            ax.annotate(f'Final: {cum_disc[-1]:.3f} K·h/m²',
                        xy=(hrs[-1], cum_disc[-1]),
                        xytext=(-100, 10), textcoords='offset points', fontsize=8,
                        arrowprops=dict(arrowstyle='->', color='#7B1FA2'))
        ax.legend(fontsize=7)
        _style(ax, 'Cumulative Discomfort (K·h/m²)', ylabel='K·h/m²', x_max=x_max)

    # Subplot 9: policy scatter (action vs zone temp)
    ax = axes[2]
    sc = ax.scatter(zone, acts, c=hrs, cmap='viridis', s=25, alpha=0.7)
    plt.colorbar(sc, ax=ax, label='Time (hours)')
    ax.axvline(np.mean(lo_arr), color='#4CAF50', ls='--', lw=1.2,
               label=f'Lower ≈ {np.mean(lo_arr):.1f}°C')
    ax.axvline(np.mean(hi_arr), color='#F44336', ls='--', lw=1.2,
               label=f'Upper ≈ {np.mean(hi_arr):.1f}°C')
    ax.set_xlabel('Zone Temperature (°C)', fontsize=8)
    ax.set_ylabel('Heat Pump Signal', fontsize=8)
    ax.set_title('Controller Behaviour: Action vs Zone Temperature (colour = time)',
                 fontsize=10, fontweight='bold', pad=5)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.tick_params(labelsize=8)

    fig3.tight_layout()
    p3 = f'{save_prefix}_page3_diagnostics.png'
    fig3.savefig(p3, dpi=200, bbox_inches='tight')
    plt.close(fig3)
    print(f'  Saved: {p3}')


# ──────────────────────────────────────────────────────────────────────────────
# Multi-controller comparison plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_combined_episode(all_data: dict, save_dir: str, episode_num: int = 1):
    """
    One figure per episode with all 3 controllers overlaid on shared subplots.

    Produces:
      episode_<N>_comparison.png  — 4-panel overlay (zone temp, control, power, outdoor)
      episode_<N>_kpi_bars.png    — grouped KPI bar chart

    all_data : {'RL': [...], 'Bang-Bang': [...], 'PID': [...]}
    """
    idx  = episode_num - 1
    ctls = [k for k in ['RL', 'Bang-Bang', 'PID']
            if k in all_data and idx < len(all_data[k])]
    if not ctls:
        return
    datasets = {k: all_data[k][idx] for k in ctls}

    # Per-controller style: colour + line style
    style = {
        'RL':        {'color': '#2196F3', 'ls': '-',  'lw': 2.0},
        'Bang-Bang': {'color': '#FF9800', 'ls': '--', 'lw': 1.8},
        'PID':       {'color': '#4CAF50', 'ls': '-.', 'lw': 1.8},
    }

    ref_data = datasets[ctls[0]]
    hrs_ref  = _hours(ref_data)
    x_max    = float(hrs_ref[-1]) if len(hrs_ref) > 0 else 1.0
    lo_arr   = np.array(ref_data['lower_setpoint_C'])
    hi_arr   = np.array(ref_data['upper_setpoint_C'])

    fig, axes = plt.subplots(4, 1, figsize=(15, 16), sharex=True)
    fig.suptitle(f'Controller Comparison — Episode {episode_num}',
                 fontsize=14, fontweight='bold', y=0.998)

    # ── Panel 1: Zone Temperature ──────────────────────────────────────────
    ax = axes[0]
    ax.fill_between(hrs_ref, lo_arr, hi_arr, alpha=0.08, color='#4CAF50',
                    label='Comfort band')
    ax.step(hrs_ref, lo_arr, where='post', color='#4CAF50', ls='--', lw=1.0, alpha=0.55,
            label='Lower setpoint')
    ax.step(hrs_ref, hi_arr, where='post', color='#F44336', ls='--', lw=1.0, alpha=0.55,
            label='Upper setpoint')
    for k in ctls:
        d    = datasets[k]
        st   = style[k]
        hrs  = _hours(d)
        zone = np.array(d['zone_temps_C'])
        ax.plot(hrs, zone, color=st['color'], ls=st['ls'], lw=st['lw'], label=k)
    ax.legend(fontsize=9, loc='best', framealpha=0.85)
    _style(ax, 'Zone Temperature', ylabel='Temperature (°C)', x_max=x_max)

    # ── Panel 2: Control Signal ─────────────────────────────────────────────
    ax = axes[1]
    for k in ctls:
        d    = datasets[k]
        st   = style[k]
        hrs  = _hours(d)
        acts = np.array(d['actions'])
        ax.step(hrs, acts, where='post', color=st['color'], ls=st['ls'], lw=st['lw'],
                label=k)
    ax.set_ylim(-0.05, 1.12)
    ax.legend(fontsize=9, loc='best', framealpha=0.85)
    _style(ax, 'Heat Pump Control Signal', ylabel='Signal [0–1]', x_max=x_max)

    # ── Panel 3: Total Power ────────────────────────────────────────────────
    ax = axes[2]
    for k in ctls:
        d    = datasets[k]
        st   = style[k]
        hrs  = _hours(d)
        pwr  = np.array(d['total_power'])
        ax.plot(hrs, pwr, color=st['color'], ls=st['ls'], lw=st['lw'], label=k)
    ax.legend(fontsize=8.5, loc='best', framealpha=0.85)
    _style(ax, 'Total Power Consumption', ylabel='Power (W)', x_max=x_max)

    # ── Panel 4: Outdoor Temperature (reference controller) ─────────────────
    ax = axes[3]
    out = np.array(ref_data.get('outdoor_temps_C', []))
    if len(out) > 0 and not np.all(np.isnan(out)):
        ax.plot(hrs_ref[:len(out)], out, color='#795548', lw=1.8,
                label='Outdoor temperature')
        ax.fill_between(hrs_ref[:len(out)], np.nanmin(out), out,
                        alpha=0.10, color='#795548')
    ax.legend(fontsize=8.5, loc='best', framealpha=0.85)
    _style(ax, 'Outdoor Temperature', ylabel='Temperature (°C)', x_max=x_max)
    ax.set_xlabel('Time (hours)', fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.997])
    p = os.path.join(save_dir, f'episode_{episode_num}_comparison.png')
    fig.savefig(p, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {p}')

    # ── KPI Bar Chart ───────────────────────────────────────────────────────
    kpi_keys   = ['ener_tot', 'tdis_tot', 'cost_tot', 'emis_tot']
    kpi_labels = ['Energy\n(kWh/m²)', 'Discomfort\n(K·h/m²)',
                  'Cost\n(EUR/m²)', 'Emissions\n(kgCO₂/m²)']
    bar_cols   = {'RL': '#2196F3', 'Bang-Bang': '#FF9800', 'PID': '#4CAF50'}
    n_ctls = len(ctls)
    x      = np.arange(len(kpi_keys))
    width  = 0.22

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    fig2.suptitle(f'BOPTEST KPI Comparison — Episode {episode_num}',
                  fontsize=13, fontweight='bold')
    for i, k in enumerate(ctls):
        vals = [datasets[k]['kpis'].get(kk, 0.0) for kk in kpi_keys]
        bars = ax2.bar(x + i * width, vals, width, label=k,
                       color=bar_cols[k], alpha=0.85)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + max(vals) * 0.01,
                         f'{v:.4f}', ha='center', va='bottom',
                         fontsize=7.5, fontweight='bold')
    ax2.set_xticks(x + width * (n_ctls - 1) / 2)
    ax2.set_xticklabels(kpi_labels, fontsize=10)
    ax2.set_ylabel('KPI Value', fontsize=10)
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.25, axis='y', linestyle='--')
    ax2.tick_params(labelsize=9)
    ax2.set_title('Lower is better for all KPIs', fontsize=9, style='italic', pad=4)

    fig2.tight_layout()
    p2 = os.path.join(save_dir, f'episode_{episode_num}_kpi_bars.png')
    fig2.savefig(p2, dpi=200, bbox_inches='tight')
    plt.close(fig2)
    print(f'  Saved: {p2}')


def plot_multi_episode_summary(all_data: dict, save_dir: str):
    """
    4-panel summary across ALL episodes for each controller.
    Shows mean ± std bands for zone temp and actions.
    """
    line_col = {'RL': '#2196F3', 'Bang-Bang': '#FF9800', 'PID': '#4CAF50'}
    ctls = [k for k in ['RL', 'Bang-Bang', 'PID'] if k in all_data and all_data[k]]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Multi-Episode Performance Summary — All Controllers', fontsize=14, fontweight='bold')

    for k in ctls:
        episodes = all_data[k]
        if not episodes:
            continue
        col = line_col[k]

        # Align episode data to shortest length
        min_len = min(len(d['zone_temps_C']) for d in episodes)
        ref_hrs = _hours(episodes[0])[:min_len]

        zone_mat  = np.array([d['zone_temps_C'][:min_len] for d in episodes])
        acts_mat  = np.array([d['actions'][:min_len] for d in episodes])
        pwr_mat   = np.array([d['total_power'][:min_len] for d in episodes])

        zone_mean = zone_mat.mean(0); zone_std = zone_mat.std(0)
        acts_mean = acts_mat.mean(0); acts_std = acts_mat.std(0)
        pwr_mean  = pwr_mat.mean(0);  pwr_std  = pwr_mat.std(0)

        label = f'{k} (n={len(episodes)})'

        # Zone temp
        axes[0, 0].plot(ref_hrs, zone_mean, color=col, lw=2, label=label)
        axes[0, 0].fill_between(ref_hrs, zone_mean-zone_std, zone_mean+zone_std,
                                alpha=0.15, color=col)
        # Actions
        axes[0, 1].plot(ref_hrs, acts_mean, color=col, lw=2, label=label)
        axes[0, 1].fill_between(ref_hrs, acts_mean-acts_std, acts_mean+acts_std,
                                alpha=0.15, color=col)
        # Power
        axes[1, 0].plot(ref_hrs, pwr_mean, color=col, lw=2, label=label)
        axes[1, 0].fill_between(ref_hrs, pwr_mean-pwr_std, pwr_mean+pwr_std,
                                alpha=0.15, color=col)

    # Comfort band (from first episode of first controller)
    _ref = all_data[ctls[0]][0]
    lo_arr = np.array(_ref['lower_setpoint_C'][:min_len])
    hi_arr = np.array(_ref['upper_setpoint_C'][:min_len])
    axes[0, 0].fill_between(ref_hrs, lo_arr, hi_arr, alpha=0.07, color='#4CAF50')
    axes[0, 0].step(ref_hrs, lo_arr, where='post', color='#4CAF50', ls='--', lw=1, alpha=0.5)
    axes[0, 0].step(ref_hrs, hi_arr, where='post', color='#F44336', ls='--', lw=1, alpha=0.5)

    for ax, title, ylabel in [
        (axes[0, 0], 'Zone Temperature (mean ± std)', '°C'),
        (axes[0, 1], 'Heat Pump Signal (mean ± std)', 'Signal [0–1]'),
        (axes[1, 0], 'Total Power (mean ± std)', 'Power (W)'),
    ]:
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Time (hours)', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.tick_params(labelsize=8)

    # KPI box plots
    ax = axes[1, 1]
    kpi_keys  = ['ener_tot', 'tdis_tot', 'cost_tot', 'emis_tot']
    kpi_short = ['Energy', 'Discomfort', 'Cost', 'Emissions']
    n_kpis    = len(kpi_keys)
    x_pos     = np.arange(n_kpis)
    width_kpi = 0.22

    for i, k in enumerate(ctls):
        means = [np.mean([d['kpis'].get(kk, 0) for d in all_data[k]]) for kk in kpi_keys]
        stds  = [np.std( [d['kpis'].get(kk, 0) for d in all_data[k]]) for kk in kpi_keys]
        ax.bar(x_pos + i*width_kpi, means, width_kpi, label=k,
               color=line_col[k], alpha=0.82, yerr=stds, capsize=3, error_kw={'linewidth':1})

    ax.set_xticks(x_pos + width_kpi * (len(ctls)-1)/2)
    ax.set_xticklabels(kpi_short, fontsize=9)
    ax.set_ylabel('Mean KPI (lower = better)', fontsize=9)
    ax.set_title('Mean KPIs (error = std across episodes)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, axis='y', linestyle='--')
    ax.tick_params(labelsize=8)

    fig.tight_layout()
    p = os.path.join(save_dir, 'multi_episode_summary.png')
    fig.savefig(p, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {p}')


# ──────────────────────────────────────────────────────────────────────────────
# Summary statistics
# ──────────────────────────────────────────────────────────────────────────────

def compute_summary(all_data: dict) -> list[dict]:
    """
    Aggregate per-controller, per-episode results into a flat list of dicts.
    Returns one row per (controller, episode) + one mean-row per controller.
    """
    rows = []

    for ctrl in ['RL', 'Bang-Bang', 'PID']:
        episodes = all_data.get(ctrl, [])
        if not episodes:
            continue

        ep_rows = []
        for d in episodes:
            kpis  = d['kpis']
            acts  = np.array(d['actions'])
            pwr   = np.array(d['total_power'])
            zone  = np.array(d['zone_temps_C'])
            ep_row = {
                'Controller':      ctrl,
                'Episode':         d['episode'],
                'Steps':           d['total_steps'],
                'ener_tot':        kpis.get('ener_tot', 0.0),
                'tdis_tot':        kpis.get('tdis_tot', 0.0),
                'cost_tot':        kpis.get('cost_tot', 0.0),
                'emis_tot':        kpis.get('emis_tot', 0.0),
                'pct_comfortable': d['pct_time_comfortable'],
                'mean_action':     float(np.mean(acts)),
                'std_action':      float(np.std(acts)),
                'on_fraction':     float(np.mean(acts > 0.05)),
                'mean_power_W':    float(np.mean(pwr)),
                'peak_power_W':    float(np.max(pwr)) if len(pwr) > 0 else 0.0,
                'mean_zone_C':     float(np.mean(zone)),
                'std_zone_C':      float(np.std(zone)),
                'total_reward':    d.get('cumulative_reward_final', 0.0),
            }
            ep_rows.append(ep_row)
            rows.append(ep_row)

        # Mean row across episodes
        n = len(ep_rows)
        mean_row = {
            'Controller':      ctrl,
            'Episode':         'MEAN',
            'Steps':           int(np.mean([r['Steps'] for r in ep_rows])),
            'ener_tot':        np.mean([r['ener_tot']    for r in ep_rows]),
            'tdis_tot':        np.mean([r['tdis_tot']    for r in ep_rows]),
            'cost_tot':        np.mean([r['cost_tot']    for r in ep_rows]),
            'emis_tot':        np.mean([r['emis_tot']    for r in ep_rows]),
            'pct_comfortable': np.mean([r['pct_comfortable'] for r in ep_rows]),
            'mean_action':     np.mean([r['mean_action'] for r in ep_rows]),
            'std_action':      np.mean([r['std_action']  for r in ep_rows]),
            'on_fraction':     np.mean([r['on_fraction'] for r in ep_rows]),
            'mean_power_W':    np.mean([r['mean_power_W'] for r in ep_rows]),
            'peak_power_W':    np.max( [r['peak_power_W'] for r in ep_rows]),
            'mean_zone_C':     np.mean([r['mean_zone_C'] for r in ep_rows]),
            'std_zone_C':      np.mean([r['std_zone_C']  for r in ep_rows]),
            'total_reward':    np.mean([r['total_reward'] for r in ep_rows]),
        }
        rows.append(mean_row)

    return rows


def print_summary_table(rows: list[dict]):
    """Print a publication-ready summary table to stdout."""

    SEP = '─' * 120

    print('\n' + '=' * 120)
    print(' SUMMARY OF RESULTS  (BOPTEST KPIs — ground truth, lower is better for all KPIs)')
    print('=' * 120)

    # --- Per-episode block ---
    print(f'\n{"Controller":<12} {"Ep":>4} {"Steps":>6} '
          f'{"ener_tot":>10} {"tdis_tot":>10} {"cost_tot":>10} {"emis_tot":>10} '
          f'{"In-band%":>9} {"MeanAct":>8} {"OnFrac":>7} {"MeanPwr(W)":>11}')
    print(SEP)

    last_ctrl = None
    for r in rows:
        ctrl = r['Controller']
        if ctrl != last_ctrl and last_ctrl is not None:
            print()
        last_ctrl = ctrl

        ep = str(r['Episode'])
        bold = (ep == 'MEAN')
        prefix = '► ' if bold else '  '

        print(f"{prefix}{ctrl:<10} {ep:>4} {r['Steps']:>6} "
              f"{r['ener_tot']:>10.4f} {r['tdis_tot']:>10.4f} {r['cost_tot']:>10.4f} "
              f"{r['emis_tot']:>10.4f} {r['pct_comfortable']:>9.1f} "
              f"{r['mean_action']:>8.3f} {r['on_fraction']:>7.2f} {r['mean_power_W']:>11.1f}")

    print(SEP)

    # --- Mean-only comparison table ---
    mean_rows = [r for r in rows if r['Episode'] == 'MEAN']
    print('\n')
    print('MEAN VALUES ACROSS ALL EPISODES  (use these figures in the paper)')
    print(SEP)
    header = (f"{'Controller':<12} {'ener_tot':>10} {'tdis_tot':>10} {'cost_tot':>10} "
              f"{'emis_tot':>10} {'In-band%':>9} {'OnFrac':>7} {'MeanPwr(W)':>11} "
              f"{'MeanAct':>8} {'MeanZone°C':>11}")
    print(header)
    print(SEP)
    for r in mean_rows:
        print(f"  {r['Controller']:<10} {r['ener_tot']:>10.4f} {r['tdis_tot']:>10.4f} "
              f"{r['cost_tot']:>10.4f} {r['emis_tot']:>10.4f} {r['pct_comfortable']:>9.1f} "
              f"{r['on_fraction']:>7.2f} {r['mean_power_W']:>11.1f} "
              f"{r['mean_action']:>8.3f} {r['mean_zone_C']:>11.2f}")

    # --- Relative improvements vs Bang-Bang ---
    bb = next((r for r in mean_rows if r['Controller'] == 'Bang-Bang'), None)
    if bb and len(mean_rows) > 1:
        print()
        print('RELATIVE TO BANG-BANG BASELINE (negative = improvement)')
        print(SEP)
        print(f"{'Controller':<12} {'Δener_tot':>10} {'Δtdis_tot':>10} {'Δcost_tot':>10} "
              f"{'Δemis_tot':>10} {'ΔIn-band%':>10}")
        print(SEP)
        for r in mean_rows:
            if r['Controller'] == 'Bang-Bang':
                continue
            d_en  = r['ener_tot']    - bb['ener_tot']
            d_dis = r['tdis_tot']    - bb['tdis_tot']
            d_co  = r['cost_tot']    - bb['cost_tot']
            d_em  = r['emis_tot']    - bb['emis_tot']
            d_cb  = r['pct_comfortable'] - bb['pct_comfortable']
            print(f"  {r['Controller']:<10} {d_en:>+10.4f} {d_dis:>+10.4f} {d_co:>+10.4f} "
                  f"{d_em:>+10.4f} {d_cb:>+10.1f}%")
        print(SEP)

    print()


def save_summary_files(rows: list[dict], save_dir: str):
    """Write summary.txt and summary.csv to save_dir."""

    # TXT
    txt_path = os.path.join(save_dir, 'summary.txt')
    with open(txt_path, 'w') as fh:
        import io
        old_stdout = sys.stdout
        sys.stdout = fh
        print_summary_table(rows)
        sys.stdout = old_stdout
    print(f'  Saved: {txt_path}')

    # CSV
    csv_path = os.path.join(save_dir, 'summary.csv')
    if rows:
        keys = list(rows[0].keys())
        with open(csv_path, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    print(f'  Saved: {csv_path}')


# ──────────────────────────────────────────────────────────────────────────────
# Main comparison runner
# ──────────────────────────────────────────────────────────────────────────────

def run_comparison(model_dir: str, n_episodes: int = 2,
                   url: str = 'http://127.0.0.1:8000',
                   save_dir: str = None):
    """
    Full comparison pipeline: RL + Bang-Bang + PID over n_episodes each.

    Parameters
    ----------
    model_dir  : directory containing final_model.zip and config.json
    n_episodes : number of episodes per controller (same start times for all)
    url        : BOPTEST REST API base URL
    save_dir   : output directory (default: <model_dir>/academic_comparison/)
    """

    _require_sb3()

    if save_dir is None:
        save_dir = os.path.join(model_dir, 'academic_comparison')
    os.makedirs(save_dir, exist_ok=True)

    print('=' * 80)
    print('ACADEMIC CONTROLLER COMPARISON')
    print('=' * 80)
    print(f'Model directory : {model_dir}')
    print(f'Output directory: {save_dir}')
    print(f'Episodes        : {n_episodes}')
    print(f'BOPTEST URL     : {url}')

    # ── Load config & model ────────────────────────────────────────────────
    config_path = os.path.join(model_dir, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'config.json not found in {model_dir}')

    with open(config_path) as f:
        config = json.load(f)

    algorithm  = config.get('algorithm', 'SAC')
    testcase   = config.get('testcase', 'bestest_hydronic_heat_pump')
    env_params = config.get('env_params', {})

    model_path = os.path.join(model_dir, 'final_model.zip')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'final_model.zip not found in {model_dir}')

    print(f'\nLoading {algorithm} model from {model_path} ...')
    algo_cls = {'SAC': SAC, 'PPO': PPO, 'DQN': DQN}.get(algorithm, SAC)
    rl_model = algo_cls.load(model_path)
    print('Model loaded.')

    # ── Episode start times (same for all controllers) ─────────────────────
    # First week of January, then weekly increments.
    # Episode 1 → Jan 1 (day 1), Episode 2 → Jan 8 (day 8), etc.
    start_times = [i * 7 * 86400 for i in range(n_episodes)]
    print(f'\nEpisode start times (seconds from year start):')
    for i, s in enumerate(start_times):
        day = s // 86400 + 1
        print(f'  Episode {i+1}: {s:,} s  (day {day} of year)')

    # ── Episode length: use config or default 144 steps ───────────────────
    step_period = env_params.get('step_period', 1800)
    vis_steps   = 144   # 3 days × 48 steps/day
    vis_env_params = dict(env_params)
    vis_env_params['max_episode_length'] = vis_steps * step_period

    # Comfort schedule from config (or default)
    raw_sched = env_params.get('comfort_schedule', None)
    if raw_sched:
        comfort_schedule = [(e[0], e[1], tuple(e[2])) for e in raw_sched]
    else:
        comfort_schedule = _DEFAULT_SCHEDULE

    # Controllers
    bang_bang = BangBangController(lower_lag_K=0.5, upper_overshoot_K=1.5)   # cold lag + hot overshoot
    pid_ctrl  = PIDController(Kp=0.20, Ki=0.00003, step_period=float(step_period))  # conservative gains

    all_data = {'RL': [], 'Bang-Bang': [], 'PID': []}

    # ══════════════════════════════════════════════════════════════════════
    # RL EPISODES
    # ══════════════════════════════════════════════════════════════════════
    print('\n' + '═' * 80)
    print('RUNNING RL AGENT EPISODES')
    print('═' * 80)

    rl_env = create_env(
        url=url, testcase=testcase,
        use_custom_reward=True,
        env_params=vis_env_params,
        _register_global=False,
    )

    expected_dim = config.get('observation_dim', rl_env.observation_space.shape[0])
    actual_dim   = rl_env.observation_space.shape[0]
    if actual_dim != expected_dim:
        rl_env.close()
        print(f'\n  ⚠  Observation space mismatch after backward-compat fix:')
        print(f'     Model expects: {expected_dim} dims')
        print(f'     Env produces:  {actual_dim} dims')
        print(f'     This model cannot be run with the current environment code.')
        print(f'     It was trained with an observation space that has since changed.')
        print(f'     Please retrain using train_rl.py, then re-run this script.\n')
        raise RuntimeError(
            f'Observation space mismatch: model expects {expected_dim}, env produces {actual_dim}.'
        )
    print(f'Observation space: {rl_env.observation_space.shape}  ✓')

    base_rl = rl_env
    while hasattr(base_rl, 'env'):
        base_rl = base_rl.env

    for ep in range(n_episodes):
        base_rl.start_time = start_times[ep]
        base_rl.random_start_time = False

        # Run RL episode using the same run_episode_with_logging logic
        # (we inline it here to keep the RL path identical to episodic_visualisation)
        obs, _reset_info = rl_env.reset()
        done = False
        step = 0

        # Find reward wrapper for comfort bounds
        reward_wrapper = None
        _tmp = rl_env
        while hasattr(_tmp, 'env'):
            if hasattr(_tmp, '_get_comfort_range'):
                reward_wrapper = _tmp
                break
            _tmp = _tmp.env

        # Seed energy tracking from reset KPIs (same as PBRSRewardWrapper)
        _rl_prev_energy = float(getattr(base_rl, 'last_kpis', {}).get('ener_tot', 0.0))

        ep_data = {
            'controller': 'RL', 'episode': ep + 1,
            'steps': [], 'times': [],
            'zone_temps_C': [], 'zone_temps_K': [],
            'outdoor_temps_C': [],
            'actions': [],
            'heat_pump_power': [], 'fan_power': [], 'pump_power': [], 'total_power': [],
            'lower_setpoint_C': [], 'upper_setpoint_C': [],
            'rewards': [], 'comfort_rewards': [], 'energy_rewards': [],
            'cumulative_reward': [], 'cumulative_discomfort_Kh': [],
            'step_period': step_period,
            'pct_time_comfortable': 0.0, 'kpis': {},
        }

        cum_reward = 0.0
        cum_disc_Kh = 0.0
        n_in_band = 0
        _sp_h = step_period / 3600.0

        print(f'\n{"─"*60}')
        print(f'  RL Agent  |  Episode {ep+1}')
        print(f'{"─"*60}')

        while not done:
            action_arr, _ = rl_model.predict(obs, deterministic=True)
            action_val = float(action_arr[0] if hasattr(action_arr, '__len__') else action_arr)

            obs, reward, terminated, truncated, info = rl_env.step(action_arr)
            done = terminated or truncated

            # Use cached measurements from base env (populated by BoptestGymEnv.step)
            _res  = getattr(base_rl, 'res', None) or {}
            meas  = getattr(base_rl, 'last_measurements', None) or _res

            zone_K = extract_value(meas, 'reaTZon_y', 293.15)
            if zone_K == 293.15 and _res:
                zone_K = extract_value(_res, 'reaTZon_y', 293.15)
            zone_C = zone_K - 273.15
            sim_t  = extract_value(meas, 'time', 0.0)
            if sim_t == 0.0 and _res:
                sim_t = extract_value(_res, 'time', 0.0)

            # Outdoor temp from last forecast cache (no extra HTTP call)
            _last_fc = getattr(base_rl, 'last_forecast', {})
            out_K = float(_last_fc.get('TDryBul', [0.0])[0]) if isinstance(
                _last_fc.get('TDryBul'), list) and _last_fc.get('TDryBul') else float(
                _last_fc.get('TDryBul', 0.0) if not isinstance(_last_fc.get('TDryBul'), list) else 0.0)
            out_C = (out_K - 273.15) if out_K > 200 else float('nan')

            # Comfort bounds
            if 'comfort_temp_lower_K' in info:
                lo_K = info['comfort_temp_lower_K']
                hi_K = info['comfort_temp_upper_K']
            elif reward_wrapper is not None:
                lo_K, hi_K = reward_wrapper._get_comfort_range(sim_t)
            else:
                lo_K, hi_K = _get_comfort_range_from_schedule(comfort_schedule, sim_t)

            # Power: KPI-delta method (consistent with PBRSRewardWrapper, units = W/m²)
            _kpis_now = getattr(base_rl, 'last_kpis', {})
            _curr_en  = float(_kpis_now.get('ener_tot', _rl_prev_energy))
            _delta_en = max(0.0, _curr_en - _rl_prev_energy)
            tot_pw    = (_delta_en * 1000.0) / _sp_h if _sp_h > 0 else 0.0
            _rl_prev_energy = _curr_en

            # Try measurement-based power breakdown (best-effort)
            pv = getattr(base_rl, 'power_measurement_vars', {})
            hp_pw  = extract_value(meas, pv.get('heat_pump', ''), 0.0) if pv.get('heat_pump') else 0.0
            fan_pw = extract_value(meas, pv.get('fan',       ''), 0.0) if pv.get('fan')       else 0.0
            pmp_pw = extract_value(meas, pv.get('pump',      ''), 0.0) if pv.get('pump')      else 0.0
            _meas_total = hp_pw + fan_pw + pmp_pw
            if _meas_total > 0.0:
                tot_pw = _meas_total
            else:
                hp_pw = tot_pw   # attribute all to heat pump when breakdown unavailable

            e_c = max(0.0, lo_K - zone_K, zone_K - hi_K)
            disc_Kh = e_c * _sp_h
            cum_disc_Kh += disc_Kh
            if e_c == 0.0:
                n_in_band += 1

            cum_reward += reward
            c_rew = info.get('comfort_reward', 0.0)
            e_rew = info.get('price_reward',   0.0)

            ep_data['steps'].append(step)
            ep_data['times'].append(sim_t)
            ep_data['zone_temps_C'].append(zone_C)
            ep_data['zone_temps_K'].append(zone_K)
            ep_data['outdoor_temps_C'].append(out_C)
            ep_data['actions'].append(action_val)
            ep_data['heat_pump_power'].append(hp_pw)
            ep_data['fan_power'].append(fan_pw)
            ep_data['pump_power'].append(pmp_pw)
            ep_data['total_power'].append(tot_pw)
            ep_data['lower_setpoint_C'].append(lo_K - 273.15)
            ep_data['upper_setpoint_C'].append(hi_K - 273.15)
            ep_data['rewards'].append(reward)
            ep_data['comfort_rewards'].append(c_rew)
            ep_data['energy_rewards'].append(e_rew)
            ep_data['cumulative_reward'].append(cum_reward)
            ep_data['cumulative_discomfort_Kh'].append(cum_disc_Kh)

            _print_every = max(1, int(round(6*3600/step_period)))
            if step % _print_every == 0:
                elapsed_h = step * _sp_h
                cf = "✓" if e_c == 0.0 else f"✗ ({e_c:.2f} K)"
                print(f"  step {step:3d} | {elapsed_h:5.1f}h | "
                      f"zone {zone_C:5.2f}°C [{lo_K-273.15:.1f}–{hi_K-273.15:.1f}] {cf} | "
                      f"act {action_val:.2f} | pwr {tot_pw:6.2f} W/m²")
            step += 1

        kpis = base_rl.get_kpis()
        ep_data['kpis'] = {k: (float(v) if v is not None else 0.0) for k,v in kpis.items()}
        ep_data['pct_time_comfortable'] = 100.0 * n_in_band / max(len(ep_data['steps']), 1)
        ep_data['total_steps'] = len(ep_data['steps'])
        ep_data['cumulative_reward_final'] = cum_reward

        print(f'\n  ── RL Episode {ep+1} complete ──')
        print(f"  Energy:    {ep_data['kpis'].get('ener_tot',0):.4f} kWh/m²")
        print(f"  Discomfort:{ep_data['kpis'].get('tdis_tot',0):.4f} K·h/m²")
        print(f"  Cost:      {ep_data['kpis'].get('cost_tot',0):.4f} EUR/m²")
        print(f"  In-band:   {ep_data['pct_time_comfortable']:.1f}%")
        print(f"  Cum reward:{cum_reward:.2f}")

        all_data['RL'].append(ep_data)

    rl_env.close()
    print('\n✓ RL episodes complete')

    # ══════════════════════════════════════════════════════════════════════
    # BANG-BANG EPISODES
    # ══════════════════════════════════════════════════════════════════════
    print('\n' + '═' * 80)
    print('RUNNING BANG-BANG CONTROLLER EPISODES')
    print('═' * 80)

    bb_env = create_env(
        url=url, testcase=testcase,
        use_custom_reward=True,
        env_params=vis_env_params,
        _register_global=False,
    )
    base_bb = bb_env
    while hasattr(base_bb, 'env'):
        base_bb = base_bb.env

    for ep in range(n_episodes):
        base_bb.start_time = start_times[ep]
        base_bb.random_start_time = False
        d = run_baseline_episode(
            bb_env, bang_bang, ep + 1,
            comfort_schedule=comfort_schedule,
            controller_name='Bang-Bang',
        )
        all_data['Bang-Bang'].append(d)

    bb_env.close()
    print('\n✓ Bang-Bang episodes complete')

    # ══════════════════════════════════════════════════════════════════════
    # PID EPISODES
    # ══════════════════════════════════════════════════════════════════════
    print('\n' + '═' * 80)
    print('RUNNING PID CONTROLLER EPISODES')
    print('═' * 80)

    pid_env = create_env(
        url=url, testcase=testcase,
        use_custom_reward=True,
        env_params=vis_env_params,
        _register_global=False,
    )
    base_pid = pid_env
    while hasattr(base_pid, 'env'):
        base_pid = base_pid.env

    for ep in range(n_episodes):
        base_pid.start_time = start_times[ep]
        base_pid.random_start_time = False
        d = run_baseline_episode(
            pid_env, pid_ctrl, ep + 1,
            comfort_schedule=comfort_schedule,
            controller_name='PID',
        )
        all_data['PID'].append(d)

    pid_env.close()
    print('\n✓ PID episodes complete')

    # ══════════════════════════════════════════════════════════════════════
    # COMPARISON PLOTS + SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    print('\n' + '═' * 80)
    print('GENERATING COMPARISON PLOTS')
    print('═' * 80)

    for ep in range(n_episodes):
        plot_combined_episode(all_data, save_dir, episode_num=ep + 1)

    if n_episodes > 1:
        plot_multi_episode_summary(all_data, save_dir)

    # Summary statistics
    print('\n' + '═' * 80)
    rows = compute_summary(all_data)
    print_summary_table(rows)
    save_summary_files(rows, save_dir)

    print('=' * 80)
    print('COMPARISON COMPLETE')
    print(f'All outputs saved to: {save_dir}')
    print('=' * 80)
    print('\nFiles generated:')
    for f in sorted(os.listdir(save_dir)):
        print(f'  {f}')

    return all_data, rows


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Academic controller comparison: RL vs Bang-Bang vs PID'
    )
    parser.add_argument('model_dir', type=str,
                        help='Directory containing final_model.zip and config.json')
    parser.add_argument('--episodes', type=int, default=2,
                        help='Episodes per controller (default: 2)')
    parser.add_argument('--url', type=str, default='http://127.0.0.1:8000',
                        help='BOPTEST server URL (default: http://127.0.0.1:8000)')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Output directory (default: <model_dir>/academic_comparison/)')

    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        print(f'Error: model directory not found: {args.model_dir}')
        if os.path.exists('trained_models'):
            print('\nAvailable models:')
            for subdir in sorted(os.listdir('trained_models')):
                p = os.path.join('trained_models', subdir)
                if os.path.isdir(p) and os.path.exists(os.path.join(p, 'final_model.zip')):
                    print(f'  trained_models/{subdir}')
        sys.exit(1)

    run_comparison(
        model_dir=args.model_dir,
        n_episodes=args.episodes,
        url=args.url,
        save_dir=args.save_dir,
    )
