"""
Environment validation script for BOPTEST Gymnasium RL.

Runs a configurable number of steps with RANDOM actions sampled from the
action space, prints a step-by-step table, and produces a PASS/FAIL
validation summary to confirm the environment is wired up correctly.

Usage:
    python validate_env.py                 # 48 steps (2 days) by default
    python validate_env.py --steps 24      # 1 day
    python validate_env.py --url http://127.0.0.1:8000
"""

import sys
import os
import argparse
import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_rl import create_env, _find_wrapper
from boptest_gym_env import BoptestGymEnv, CustomRewardWrapper, extract_value


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fetch_outdoor_temp_C(base_env):
    """Return current outdoor dry-bulb temperature in °C from the forecast endpoint.

    Returns float('nan') if the fetch fails so the table still prints cleanly.
    """
    try:
        resp = requests.put(
            f'{base_env.url}/forecast/{base_env.testid}',
            json={
                'point_names': ['TDryBul'],
                'horizon': int(base_env.step_period),
                'interval': int(base_env.step_period),
            },
            timeout=10,
        )
        if resp.status_code == 200:
            vals = resp.json().get('payload', {}).get('TDryBul', [])
            temp_K = float(vals[0]) if isinstance(vals, list) and vals else float(vals or 0)
            if temp_K > 200:
                return temp_K - 273.15
        # Fallback: plain GET (no point_names filter)
        resp = requests.get(f'{base_env.url}/forecast/{base_env.testid}', timeout=10)
        if resp.status_code == 200:
            vals = resp.json().get('payload', {}).get('TDryBul', [])
            temp_K = float(vals[0]) if isinstance(vals, list) and vals else float(vals or 0)
            if temp_K > 200:
                return temp_K - 273.15
    except Exception:
        pass
    return float('nan')


def _zone_temp_from_info_or_res(info, base_env):
    """Pull zone temperature (°C) from the info dict; fall back to base_env.res."""
    temp_K = info.get('zone_temp')
    if temp_K is None or temp_K == 293.15:
        # Try the advance response directly (always value-format)
        res = getattr(base_env, 'res', {}) or {}
        from_res = extract_value(res, 'reaTZon_y', 293.15)
        if from_res != 293.15:
            temp_K = from_res
    if temp_K is None:
        temp_K = 293.15
    return float(temp_K) - 273.15


# ── Main validation routine ────────────────────────────────────────────────────

def validate(url='http://127.0.0.1:8000', n_steps=48):
    """Run n_steps of random actions and print validation results.

    Returns True when all checks pass, False otherwise.
    """

    SEP = '=' * 100
    sep = '-' * 100

    print(SEP)
    print('BOPTEST ENVIRONMENT VALIDATION')
    print(SEP)
    print(f'Server URL : {url}')
    print(f'Steps      : {n_steps}')
    print()

    # ── Create environment ─────────────────────────────────────────────────────
    print('Creating environment ...')
    try:
        env = create_env(url=url, use_custom_reward=True)
    except Exception as exc:
        print(f'  FAIL  Could not create environment: {exc}')
        return False

    base_env    = _find_wrapper(env, BoptestGymEnv)
    rwd_wrapper = _find_wrapper(env, CustomRewardWrapper)

    step_h = base_env.step_period // 3600          # hours per step
    print(f'  OK   Environment created')
    print(f'       Testcase    : {base_env.testcase}')
    print(f'       Action space: {env.action_space}')
    print(f'       Obs shape   : {env.observation_space.shape}')
    print(f'       Step period : {base_env.step_period}s  ({step_h}h)')
    print(f'       Power vars  : {base_env.power_measurement_vars or "NONE DETECTED"}')
    if not base_env.power_measurement_vars:
        print('       WARNING: no power measurement variables were detected!')
        print('                Power will show as 0 W for every step.')
    print()

    # ── Reset ─────────────────────────────────────────────────────────────────
    print('Resetting environment ...')
    try:
        obs, _ = env.reset()
    except Exception as exc:
        print(f'  FAIL  reset() raised: {exc}')
        env.close()
        return False
    print(f'  OK   Reset successful  (obs shape {obs.shape})')
    print()

    # ── Comfort schedule info ──────────────────────────────────────────────────
    if rwd_wrapper is not None:
        print('Comfort schedule (dynamic day/night):')
        for start_h, end_h, (lo, hi) in rwd_wrapper.comfort_schedule:
            print(f'  {start_h:02d}:00 – {end_h:02d}:00 →  '
                  f'{lo - 273.15:.1f}°C – {hi - 273.15:.1f}°C')
        print()

    # ── Column header ──────────────────────────────────────────────────────────
    HDR = (
        f"{'Step':>5}  {'SimTime':>8}  "
        f"{'Zone(°C)':>10}  {'Band(°C)':>14}  "
        f"{'Out(°C)':>8}  "
        f"{'Action':>7}  {'Power(W)':>9}  "
        f"{'Reward':>9}  {'Comfort':>9}  {'Energy':>9}"
    )
    print(HDR)
    print(sep)

    # ── Tracking lists for validation checks ───────────────────────────────────
    zone_temps   = []
    out_temps    = []
    powers       = []
    rewards      = []
    actions_log  = []
    times_s      = []

    done = False
    step = 0

    while not done and step < n_steps:
        # ── Random action ──────────────────────────────────────────────────────
        action = env.action_space.sample()
        act_val = float(action.flat[0]) if hasattr(action, 'flat') else float(action)

        # ── Step ───────────────────────────────────────────────────────────────
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # ── Extract readings ───────────────────────────────────────────────────
        sim_time_s  = float(info.get('time', step * base_env.step_period))
        zone_temp_C = _zone_temp_from_info_or_res(info, base_env)
        total_power = float(info.get('total_power', 0.0))
        c_rwd       = float(info.get('comfort_reward', float('nan')))
        e_rwd       = float(info.get('price_reward',   float('nan')))

        # Outdoor temperature (one extra API call, acceptable for validation)
        out_C = _fetch_outdoor_temp_C(base_env)

        # Comfort bounds for this timestep
        if rwd_wrapper is not None:
            lo_K, hi_K = rwd_wrapper._get_comfort_range(sim_time_s)
            lo_C, hi_C = lo_K - 273.15, hi_K - 273.15
            in_band     = lo_C <= zone_temp_C <= hi_C
            comfort_str = f'✓ {lo_C:.0f}–{hi_C:.0f}'
            comfort_col = f'{"IN" if in_band else "OUT":>3} {lo_C:.0f}–{hi_C:.0f}'
        else:
            comfort_col = '    —'

        out_str = f'{out_C:8.2f}' if not np.isnan(out_C) else '     N/A'

        # ── Print row ──────────────────────────────────────────────────────────
        sim_label = f'{int(sim_time_s // 3600):4d}h'
        print(
            f'{step:5d}  {sim_label:>8}  '
            f'{zone_temp_C:9.3f}°  {comfort_col:>14}  '
            f'{out_str}  '
            f'{act_val:7.4f}  {total_power:9.1f}  '
            f'{reward:9.4f}  {c_rwd:9.4f}  {e_rwd:9.4f}'
        )

        # Accumulate for checks
        zone_temps.append(zone_temp_C)
        out_temps.append(out_C)
        powers.append(total_power)
        rewards.append(reward)
        actions_log.append(act_val)
        times_s.append(sim_time_s)

        step += 1

    print(sep)
    print(f'  Ran {step} steps.')
    print()

    # ── BOPTEST KPIs ───────────────────────────────────────────────────────────
    kpis = base_env.get_kpis()
    print(SEP)
    print('BOPTEST KPIs (cumulative over episode so far)')
    print(SEP)
    print(f'  Energy     : {kpis.get("ener_tot", 0):.6f}  kWh/m²')
    print(f'  Discomfort : {kpis.get("tdis_tot", 0):.6f}  K·h/zone')
    print(f'  Cost       : {kpis.get("cost_tot", 0):.6f}  EUR/m²')
    print(f'  Emissions  : {kpis.get("emis_tot", 0):.6f}  kg CO₂/m²')
    print()

    # ── Validation checks ──────────────────────────────────────────────────────
    print(SEP)
    print('VALIDATION CHECKS')
    print(SEP)

    zone_arr = np.array(zone_temps)
    rwd_arr  = np.array(rewards)
    act_arr  = np.array(actions_log)
    out_arr  = np.array(out_temps)
    pwr_arr  = np.array(powers)

    checks = []

    # 1. Zone temperature is not stuck at fallback (293.15 K = 20°C)
    all_fallback = np.all(np.abs(zone_arr - 20.0) < 0.005)
    checks.append((
        not all_fallback,
        'Zone temperature is NOT stuck at the 20°C fallback',
        f'range {zone_arr.min():.3f}°C – {zone_arr.max():.3f}°C'
        if not all_fallback else
        'ALL steps returned exactly 20.000°C  →  measurement fallback is active!'
    ))

    # 2. Zone temperature actually changes
    temp_delta = zone_arr.max() - zone_arr.min()
    checks.append((
        temp_delta > 0.01,
        'Zone temperature changes by > 0.01°C  (simulation is advancing)',
        f'Δ = {temp_delta:.4f}°C'
    ))

    # 3. Power is non-zero for steps where a large action was taken
    big_action_idx = [i for i, a in enumerate(actions_log) if a > 0.15]
    if big_action_idx:
        pwr_at_big = [powers[i] for i in big_action_idx]
        n_nonzero  = sum(1 for p in pwr_at_big if p > 0.5)
        checks.append((
            n_nonzero > 0,
            'Power > 0 W on at least one step where action > 0.15',
            f'{n_nonzero}/{len(big_action_idx)} steps with action>0.15 had measurable power'
        ))
    else:
        checks.append((
            None,
            'Power check SKIPPED  (no sampled actions exceeded 0.15)',
            f'action range was [{act_arr.min():.4f}, {act_arr.max():.4f}]'
            '  —  try more steps'
        ))

    # 4. Time advances by step_period each step
    if len(times_s) >= 2:
        deltas   = np.diff(times_s)
        all_ok   = np.all(np.abs(deltas - base_env.step_period) < 2)
        checks.append((
            all_ok,
            'Simulation time advances by step_period on every step',
            f'expected Δt={base_env.step_period}s, '
            f'observed [{deltas.min():.0f}s, {deltas.max():.0f}s]'
        ))

    # 5. Rewards respond to different actions
    checks.append((
        rwd_arr.std() > 1e-6,
        'Rewards are not all identical  (reward signal is responsive)',
        f'mean={rwd_arr.mean():.4f}  std={rwd_arr.std():.4f}  '
        f'range [{rwd_arr.min():.4f}, {rwd_arr.max():.4f}]'
    ))

    # 6. Outdoor temperature is available
    out_valid = ~np.isnan(out_arr)
    checks.append((
        out_valid.any(),
        'Outdoor temperature is available from the forecast endpoint',
        f'mean={np.nanmean(out_arr):.2f}°C  '
        f'range [{np.nanmin(out_arr):.2f}, {np.nanmax(out_arr):.2f}]°C'
        if out_valid.any() else
        'ALL steps returned NaN  →  PUT /forecast is failing (check server logs)'
    ))

    # 7. Activation signal causes power (env sends oveHeaPumY_activate=1)
    acts_source = inspect_activation_signal()
    checks.append((
        acts_source,
        "step() sends 'oveHeaPumY_activate=1' alongside the action value",
        'confirmed in BoptestGymEnv.step() source'
        if acts_source else
        'MISSING  →  heat pump override will be ignored by BOPTEST!'
    ))

    # 8. Action space bounds respected
    lo = float(env.action_space.low.flat[0])
    hi = float(env.action_space.high.flat[0])
    bounds_ok = bool(act_arr.min() >= lo - 1e-6 and act_arr.max() <= hi + 1e-6)
    checks.append((
        bounds_ok,
        'All sampled actions are within action space bounds',
        f'space [{lo:.2f}, {hi:.2f}]  '
        f'sampled [{act_arr.min():.4f}, {act_arr.max():.4f}]'
    ))

    # ── Print results ──────────────────────────────────────────────────────────
    n_pass = n_fail = n_skip = 0
    for passed, label, detail in checks:
        if passed is None:
            icon = '⚠ SKIP'
            n_skip += 1
        elif passed:
            icon = '✓ PASS'
            n_pass += 1
        else:
            icon = '✗ FAIL'
            n_fail += 1
        print(f'  {icon}  {label}')
        print(f'         {detail}')
        print()

    print(sep)
    print(f'  {n_pass} passed   {n_fail} failed   {n_skip} skipped')
    if n_fail == 0:
        print('\n  ✓  ENVIRONMENT IS WORKING CORRECTLY')
    else:
        print('\n  ✗  ENVIRONMENT HAS ISSUES — see failures above')
    print(SEP)

    env.close()
    return n_fail == 0


def inspect_activation_signal():
    """Return True if BoptestGymEnv.step() sends oveHeaPumY_activate=1."""
    import inspect
    from boptest_gym_env import BoptestGymEnv
    src = inspect.getsource(BoptestGymEnv.step)
    return '_activate' in src and 'activate_name' in src


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Validate the BOPTEST Gymnasium RL environment with random actions'
    )
    parser.add_argument(
        '--url', default='http://127.0.0.1:8000',
        help='BOPTEST server URL  (default: http://127.0.0.1:8000)'
    )
    parser.add_argument(
        '--steps', type=int, default=48,
        help='Number of random steps to run  (default: 48 = 2 days at 1h/step)'
    )
    args = parser.parse_args()

    ok = validate(url=args.url, n_steps=args.steps)
    sys.exit(0 if ok else 1)
