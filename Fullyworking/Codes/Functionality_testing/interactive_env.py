"""
Interactive environment testing script for BOPTEST Gymnasium RL.

Allows you to manually enter the heat-pump action at every step so you
can observe exactly how the reward function responds to your decisions.
The current state (zone temperature, comfort band, outdoor temperature)
is printed before each prompt so you can make an informed choice.

Output mirrors validate_env.py with full reward breakdown:
    Reward = Comfort reward + Power reward

Usage:
    python interactive_env.py                 # 48 steps (2 days) by default
    python interactive_env.py --steps 24      # 1 day
    python interactive_env.py --url http://127.0.0.1:8000
"""

import sys
import os
import argparse
import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_rl import create_env, _find_wrapper
from boptest_gym_env import BoptestGymEnv, CustomRewardWrapper, extract_value
from validate_env import _fetch_outdoor_temp_C, _zone_temp_from_info_or_res


# ── Helpers ────────────────────────────────────────────────────────────────────

def _prompt_action(action_low, action_high, last_action=None):
    """Prompt the user to type an action.

    Returns the float value entered, or None if the user quits.
    Pressing Enter without typing repeats the previous action (default 0.0
    on the first step).  'q', 'quit', or Ctrl-C all exit gracefully.
    """
    default_val = last_action if last_action is not None else action_low
    prompt = (
        f'  Enter action [{action_low:.2f}–{action_high:.2f}]'
        f'  (Enter = {default_val:.4f},  q = quit): '
    )
    while True:
        try:
            raw = input(prompt).strip()
        except (KeyboardInterrupt, EOFError):
            return None
        if raw.lower() in ('q', 'quit', 'exit'):
            return None
        if raw == '':
            return float(default_val)
        try:
            val = float(raw)
            if action_low - 1e-6 <= val <= action_high + 1e-6:
                return float(np.clip(val, action_low, action_high))
            print(f'  ⚠  {val:.4f} is outside [{action_low:.2f}, {action_high:.2f}]. Try again.')
        except ValueError:
            print(f'  ⚠  "{raw}" is not a valid number. Enter a float like 0.0, 0.5 or 1.0.')


def _state_panel(step, zone_temp_C, sim_time_s, rwd_wrapper,
                 out_C, last_action, last_power,
                 last_reward=None, last_c_rwd=None, last_p_rwd=None,
                 last_thermal_ok=None, last_price=None,
                 sep_char='─', width=80):
    """Print a state summary panel before the action prompt.

    Shows the zone temperature, comfort band status, outdoor temperature,
    and (from step 1 onward) the previous step's reward breakdown and
    action/power draw.
    """
    if rwd_wrapper is not None:
        lo_K, hi_K = rwd_wrapper._get_comfort_range(sim_time_s)
        lo_C, hi_C = lo_K - 273.15, hi_K - 273.15
        in_band    = lo_C <= zone_temp_C <= hi_C
        band_str   = f'{lo_C:.0f}–{hi_C:.0f}°C'
        status     = '✓ IN ' if in_band else '✗ OUT'
    else:
        band_str = 'N/A'
        status   = '   '

    hour_of_day = int((sim_time_s % 86400) / 3600)
    day_num     = int(sim_time_s // 86400) + 1
    out_str     = f'{out_C:.1f}°C' if not np.isnan(out_C) else 'N/A'

    print(f'\n{sep_char * width}')
    print(
        f'  Step {step:3d}  |  Day {day_num}  {hour_of_day:02d}:00'
        f'  |  Zone: {zone_temp_C:6.2f}°C  [{status}  Band: {band_str}]'
        f'  |  Outdoor: {out_str}'
    )
    if last_action is not None:
        print(
            f'  Prev action: {last_action:.4f}  |  Power: {last_power:.2f} W/m²'
        )
    if last_reward is not None:
        in_band_str = 'YES' if last_thermal_ok else 'NO'
        price_str = f'{last_price:.4f}' if last_price is not None and not np.isnan(last_price) else 'N/A'
        print(
            f'  *** LAST REWARD: {last_reward:+.4f}  '
            f'(Comfort: {last_c_rwd:+.4f}  |  PriceRwd: {last_p_rwd:+.4f}'
            f'  |  Price: {price_str} EUR/kWh  |  InBand: {in_band_str}) ***'
        )


# ── Main routine ───────────────────────────────────────────────────────────────

def run_interactive(url='http://127.0.0.1:8000', max_steps=48):
    """Run the environment with manual action input at each step.

    Returns True when the session completes without errors.
    """
    SEP   = '=' * 80
    sep   = '-' * 80
    WIDTH = 80

    print(SEP)
    print('BOPTEST INTERACTIVE ENVIRONMENT TEST')
    print(SEP)
    print(f'Server URL : {url}')
    print(f'Max steps  : {max_steps}')
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
    step_h      = base_env.step_period // 3600
    action_low  = float(env.action_space.low.flat[0])
    action_high = float(env.action_space.high.flat[0])

    print(f'  OK   Environment created')
    print(f'       Testcase      : {base_env.testcase}')
    print(f'       Action space  : [{action_low:.2f}, {action_high:.2f}]')
    print(f'       Obs shape     : {env.observation_space.shape}')
    print(f'       Step period   : {base_env.step_period}s  ({step_h}h/step)')
    if rwd_wrapper is not None:
        print(
            f'       Reward params : comfort_weight={rwd_wrapper.comfort_weight}'
            f'  price_weight={rwd_wrapper.price_weight}'
            f'  max_pen={rwd_wrapper.max_pen}'
        )
    print()

    # ── Comfort schedule ───────────────────────────────────────────────────────
    if rwd_wrapper is not None:
        print('Comfort schedule (day/night):')
        for start_h, end_h, (lo, hi) in rwd_wrapper.comfort_schedule:
            print(f'  {start_h:02d}:00 – {end_h:02d}:00  →  '
                  f'{lo - 273.15:.1f}°C – {hi - 273.15:.1f}°C')
        print()

    # ── Reset ──────────────────────────────────────────────────────────────────
    print('Resetting environment ...')
    try:
        obs, _ = env.reset()
    except Exception as exc:
        print(f'  FAIL  reset() raised: {exc}')
        env.close()
        return False
    print(f'  OK   Reset successful  (obs shape {obs.shape})')
    print()

    # Initial state for the first state panel (from reset payload via base_env.res)
    curr_zone_C = _zone_temp_from_info_or_res({}, base_env)
    curr_time_s = float(extract_value(getattr(base_env, 'res', {}) or {}, 'time', 0))
    curr_out_C  = _fetch_outdoor_temp_C(base_env)

    # ── Column header (two lines, fits in 80-char terminal) ───────────────────
    # Line 1: state + action + power   (~73 chars)
    # Line 2: reward breakdown         (~73 chars, indented)
    HDR1 = (
        f"{'Step':>5}  {'SimTime':>8}  "
        f"{'Zone(°C)':>10}  {'Band(°C)':>14}  "
        f"{'Out(°C)':>8}  "
        f"{'Action':>7}  {'Power(W/m²)':>11}  {'Price(EUR/kWh)':>14}"
    )
    HDR2 = (
        f"       "
        f"{'Reward':>13}   {'Comfort':>14}   {'PriceRwd':>14}   {'InBand':>6}"
    )
    print(HDR1)
    print(HDR2)
    print(sep)

    # ── Tracking ───────────────────────────────────────────────────────────────
    zone_temps   = []
    powers       = []
    rewards      = []
    c_rwds       = []   # comfort reward per step
    p_rwds       = []   # price reward per step
    prices       = []   # electricity price (EUR/kWh) per step
    actions_log  = []
    times_s      = []
    in_band_log  = []

    last_action     = None
    last_power      = 0.0
    last_reward     = None
    last_c_rwd      = None
    last_p_rwd      = None
    last_thermal_ok = None
    last_price      = float('nan')
    step            = 0
    done            = False

    while not done and step < max_steps:

        # ── Show current state before prompting ────────────────────────────────
        _state_panel(
            step, curr_zone_C, curr_time_s, rwd_wrapper,
            curr_out_C, last_action, last_power,
            last_reward=last_reward, last_c_rwd=last_c_rwd,
            last_p_rwd=last_p_rwd, last_thermal_ok=last_thermal_ok,
            last_price=last_price,
            width=WIDTH,
        )

        # ── Get action from user ───────────────────────────────────────────────
        action_val = _prompt_action(action_low, action_high, last_action)

        if action_val is None:
            print('\n  Session ended by user.')
            break

        last_action = action_val
        action      = np.array([action_val], dtype=np.float32)

        # ── Step ───────────────────────────────────────────────────────────────
        try:
            obs, reward, terminated, truncated, info = env.step(action)
        except Exception as exc:
            print(f'\n  ERROR during step {step}: {exc}')
            break

        done = terminated or truncated

        # ── Extract results ────────────────────────────────────────────────────
        sim_time_s        = float(info.get('time', (step + 1) * base_env.step_period))
        zone_temp_C       = _zone_temp_from_info_or_res(info, base_env)
        total_power       = float(info.get('total_power', 0.0))
        c_rwd             = float(info.get('comfort_reward',    float('nan')))
        p_rwd             = float(info.get('price_reward',      float('nan')))
        electricity_price = float(info.get('electricity_price', float('nan')))
        thermal_ok        = bool(info.get('thermal_comfort', False))
        out_C             = _fetch_outdoor_temp_C(base_env)

        # Comfort band column text
        if rwd_wrapper is not None:
            lo_K, hi_K  = rwd_wrapper._get_comfort_range(sim_time_s)
            lo_C, hi_C  = lo_K - 273.15, hi_K - 273.15
            band_col    = f'{"IN" if thermal_ok else "OUT":>3} {lo_C:.0f}–{hi_C:.0f}'
        else:
            band_col = '       —'

        out_col     = f'{out_C:8.2f}' if not np.isnan(out_C) else '     N/A'
        sim_label   = f'{int(sim_time_s // 3600):4d}h'
        in_band_str = 'YES' if thermal_ok else 'NO'

        # ── Print result: two lines per step ───────────────────────────────────
        price_col = f'{electricity_price:.4f}' if not np.isnan(electricity_price) else '           N/A'
        # Line 1: state + action + power + price
        print(
            f'{step:5d}  {sim_label:>8}  '
            f'{zone_temp_C:9.3f}°  {band_col:>14}  '
            f'{out_col}  '
            f'{action_val:7.4f}  {total_power:11.2f}  {price_col:>14}'
        )
        # Line 2: reward breakdown (indented, always visible)
        print(
            f'       '
            f'Reward: {reward:+.4f}   '
            f'Comfort: {c_rwd:+.4f}   '
            f'PriceRwd: {p_rwd:+.4f}   '
            f'InBand: {in_band_str}'
        )

        # Update state for next iteration's state panel
        curr_zone_C     = zone_temp_C
        curr_time_s     = sim_time_s
        curr_out_C      = out_C
        last_power      = total_power
        last_reward     = reward
        last_c_rwd      = c_rwd
        last_p_rwd      = p_rwd
        last_thermal_ok = thermal_ok
        last_price      = electricity_price

        # Accumulate tracking data
        zone_temps.append(zone_temp_C)
        powers.append(total_power)
        rewards.append(reward)
        c_rwds.append(c_rwd)
        p_rwds.append(p_rwd)
        prices.append(electricity_price)
        actions_log.append(action_val)
        times_s.append(sim_time_s)
        in_band_log.append(thermal_ok)

        step += 1

    print(sep)
    print(f'\n  Completed {step} step(s).')
    print()

    if not rewards:
        env.close()
        return True

    # ── Episode summary ────────────────────────────────────────────────────────
    rwd_arr  = np.array(rewards)
    pwr_arr  = np.array(powers)
    zone_arr = np.array(zone_temps)
    act_arr  = np.array(actions_log)

    in_band_steps = sum(in_band_log)
    pct_comfort   = 100.0 * in_band_steps / step if step > 0 else 0.0

    print(SEP)
    print('EPISODE SUMMARY')
    print(SEP)
    print(f'  Steps completed   : {step}')
    print(f'  Total reward      : {rwd_arr.sum():.4f}')
    print(f'  Mean reward/step  : {rwd_arr.mean():.4f}')
    print(f'  Reward range      : [{rwd_arr.min():.4f}, {rwd_arr.max():.4f}]')
    print()
    print(f'  Zone temp range   : {zone_arr.min():.2f}°C – {zone_arr.max():.2f}°C')
    print(f'  Steps in comfort  : {in_band_steps}/{step}  ({pct_comfort:.1f}%)')
    print(f'  Mean power        : {pwr_arr.mean():.2f} W/m²')
    print(f'  Power range       : [{pwr_arr.min():.2f}, {pwr_arr.max():.2f}] W/m²')
    print(f'  Action range used : [{act_arr.min():.4f}, {act_arr.max():.4f}]')
    print()

    # Reward component breakdown (from tracked per-step values)
    c_arr     = np.array(c_rwds)
    p_arr     = np.array(p_rwds)
    price_arr = np.array([x for x in prices if not np.isnan(x)], dtype=float)
    print(f'  Comfort component : mean={c_arr.mean():.4f}  '
          f'range [{c_arr.min():.4f}, {c_arr.max():.4f}]')
    print(f'  Price component   : mean={p_arr.mean():.4f}  '
          f'range [{p_arr.min():.4f}, {p_arr.max():.4f}]')
    if price_arr.size > 0:
        print(f'  Electricity price : mean={price_arr.mean():.4f}  '
              f'range [{price_arr.min():.4f}, {price_arr.max():.4f}]  EUR/kWh')
    else:
        print(f'  Electricity price : N/A (no price data)')
    print()

    # ── BOPTEST KPIs ──────────────────────────────────────────────────────────
    kpis = base_env.get_kpis()
    print(SEP)
    print('BOPTEST KPIs (cumulative over episode)')
    print(SEP)
    print(f'  Energy     : {kpis.get("ener_tot", 0):.6f}  kWh/m²')
    print(f'  Discomfort : {kpis.get("tdis_tot", 0):.6f}  K·h/zone')
    print(f'  Cost       : {kpis.get("cost_tot", 0):.6f}  EUR/m²')
    print(f'  Emissions  : {kpis.get("emis_tot", 0):.6f}  kg CO₂/m²')
    print(SEP)

    env.close()
    return True


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Interactive BOPTEST environment test with manual action input'
    )
    parser.add_argument(
        '--url', default='http://127.0.0.1:8000',
        help='BOPTEST server URL  (default: http://127.0.0.1:8000)'
    )
    parser.add_argument(
        '--steps', type=int, default=48,
        help='Maximum number of steps  (default: 48 = 2 days at 1h/step)'
    )
    args = parser.parse_args()

    run_interactive(url=args.url, max_steps=args.steps)
