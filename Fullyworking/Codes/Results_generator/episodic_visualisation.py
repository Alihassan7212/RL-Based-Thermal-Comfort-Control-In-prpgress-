"""
Detailed Episode Visualization for Trained BOPTEST RL Models

Produces 3 separate figure pages per episode:
  Page 1 — Temperature & Comfort
    1. Internal (zone) temperature vs time  +  comfort shading
    2. External (outdoor) temperature vs time
    3. Per-step discomfort penalty + cumulative discomfort

  Page 2 — Actions & Power
    4. Heat pump control action overlaid with zone temperature
    5. Total power + per-component breakdown
    6. Electricity price vs action  (price-responsiveness check)

  Page 3 — Reward & Diagnostics
    7. Per-step reward (stacked comfort + energy components)
    8. Cumulative reward accumulated over episode
    9. Policy scatter: action vs zone temperature  (what the model learned)
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import json

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from stable_baselines3 import PPO, DQN, SAC
except ModuleNotFoundError:
    PPO = DQN = SAC = None
from train_rl import create_env
import requests


def _require_sb3():
    if any(v is None for v in (PPO, DQN, SAC)):
        raise RuntimeError(
            "stable_baselines3 is required to load trained RL models. "
            "Install it in the active Python environment."
        )


def extract_value(data, key, default=0):
    """Extract value from BOPTEST measurement dict."""
    val = data.get(key, default)
    if isinstance(val, dict):
        return float(val.get('value', default))
    return float(val)


def run_episode_with_logging(env, model, episode_num=1):
    """
    Run one episode and log all data for visualization.
    
    Returns
    -------
    episode_data : dict
        Dictionary containing all logged information
    """
    
    obs, _ = env.reset()
    done = False
    step = 0
    
    # Retrieve comfort bounds and schedule from the wrapper (works for any config)
    # FIX B16: CustomRewardWrapper now has comfort_schedule and _get_comfort_range;
    # look for the wrapper that has comfort_temp_range (CustomRewardWrapper).
    comfort_low_C, comfort_high_C = 22.0, 24.0
    reward_wrapper = None
    temp_env = env
    while hasattr(temp_env, 'env'):
        if hasattr(temp_env, 'comfort_temp_range'):
            comfort_low_C  = temp_env.comfort_temp_range[0] - 273.15
            comfort_high_C = temp_env.comfort_temp_range[1] - 273.15
            # If this wrapper also has _get_comfort_range it's CustomRewardWrapper
            if hasattr(temp_env, '_get_comfort_range'):
                reward_wrapper = temp_env
            break
        temp_env = temp_env.env
    
    # Storage for episode data
    data = {
        'steps': [],
        'times': [],
        'zone_temps': [],
        'zone_temps_K': [],
        'outdoor_temps': [],
        'actions': [],
        'heat_pump_power': [],
        'fan_power': [],
        'pump_power': [],
        'total_power': [],
        'solar_radiation': [],
        'internal_gains': [],
        'electricity_price': [],
        'lower_setpoint': [],
        'upper_setpoint': [],
        'rewards': [],
        'comfort_rewards': [],
        'price_rewards': [],
        'cumulative_reward': [],
        'cumulative_energy': [],
        'cumulative_discomfort': [],
        'comfort_low_C':  comfort_low_C,
        'comfort_high_C': comfort_high_C,
        'lower_setpoint_comfort': [],
        'upper_setpoint_comfort': [],
        'step_period': 1800,   # filled in from base_env below; used for x-axis conversion
    }
    
    print(f"\n{'='*60}")
    print(f"Running Episode {episode_num}")
    print(f"{'='*60}")
    
    cumulative_reward = 0
    cumulative_energy = 0
    cumulative_discomfort = 0
    
    while not done:
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Get detailed measurements from BOPTEST
        # Navigate through wrappers to get base environment
        base_env = env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        
        response = requests.get(f'{base_env.url}/measurements/{base_env.testid}')
        if response.status_code == 200:
            measurements = response.json()['payload']

            # B26 fix: GET /measurements may return metadata-format dicts
            # (Minimum/Maximum/unit but no 'value') for some or all variables.
            # base_env.res (from the most recent POST /advance) is always in
            # value format.  Patch any metadata-format or missing entries so
            # that extract_value() returns actual readings, not the fallback 0.
            _base_res = getattr(base_env, 'res', None) or {}
            for _var, _val in list(measurements.items()):
                _is_meta = (
                    isinstance(_val, dict)
                    and 'value' not in _val
                    and any(k in _val for k in ('Minimum', 'Maximum', 'Unit', 'unit'))
                )
                if (_val is None or _is_meta) and _var in _base_res:
                    measurements[_var] = _base_res[_var]

            # Fetch forecast vars — these are predictive variables in BOPTEST and
            # return 0 (or stale values) from the /measurements/ endpoint.
            # LowerSetp / UpperSetp are forecast vars (no [1] suffix for
            # single-zone testcases like bestest_hydronic_heat_pump).
            # We filter by variables that actually exist in this testcase so
            # that an unknown name never causes the whole request to fail.
            _all_fc_vars = getattr(base_env, 'all_predictive_vars', {})
            _desired_vars = [
                'TDryBul', 'HDirNor', 'PriceElectricPowerHighlyDynamic',
                'LowerSetp', 'UpperSetp',
            ]
            forecast_vars = [v for v in _desired_vars if v in _all_fc_vars] or _desired_vars
            fc_values = {v: None for v in _desired_vars}

            def _parse_fc_payload(payload):
                for var in _desired_vars:
                    vals = payload.get(var, [])
                    if isinstance(vals, list) and vals:
                        fc_values[var] = float(vals[0])
                    elif isinstance(vals, (int, float)):
                        fc_values[var] = float(vals)

            try:
                # Preferred: PUT with point_names filter (BOPTEST v2)
                fc_resp = requests.put(
                    f'{base_env.url}/forecast/{base_env.testid}',
                    json={'point_names': forecast_vars,
                          'horizon': int(base_env.step_period),
                          'interval': int(base_env.step_period)},
                    timeout=10
                )
                if fc_resp.status_code == 200:
                    _parse_fc_payload(fc_resp.json().get('payload', {}))
                else:
                    # Fallback: GET all forecast variables
                    fc_get = requests.get(
                        f'{base_env.url}/forecast/{base_env.testid}',
                        timeout=10
                    )
                    if fc_get.status_code == 200:
                        _parse_fc_payload(fc_get.json().get('payload', {}))
            except Exception:
                pass

            # Replace None with 0.0 so arithmetic below never fails
            fc_values = {v: (fc_values[v] if fc_values[v] is not None else 0.0)
                         for v in _desired_vars}

            # Power variable names — detected during env init; resolved here so
            # the diagnostic block below can reference pv before power values
            # are actually read.
            pv = getattr(base_env, 'power_measurement_vars', {})

            # Capture step_period so plot_episode_detailed can convert steps → hours
            data['step_period'] = getattr(base_env, 'step_period', 1800)

            # Extract all measurements (already patched above by B26 supplement)
            zone_temp_K = extract_value(measurements, 'reaTZon_y', 293.15)
            zone_temp_C = zone_temp_K - 273.15

            # ─── Step-0 diagnostic: show raw BOPTEST values ───────────────────
            # This helps identify whether reaTZon_y is being read correctly or
            # falling back to the 293.15 K (20 °C) default.
            if step == 0:
                _raw_zone = measurements.get('reaTZon_y', 'MISSING')
                _hp_var = pv.get('heat_pump', 'N/A')
                _raw_hp = measurements.get(_hp_var, 'MISSING') if _hp_var != 'N/A' else 'N/A'
                _res_zone = (getattr(base_env, 'res', {}) or {}).get('reaTZon_y', 'NOT IN res')
                print(f"\n[DIAGNOSTIC step 0]")
                print(f"  GET /measurements reaTZon_y  = {_raw_zone!r}")
                print(f"  base_env.res     reaTZon_y  = {_res_zone!r}")
                print(f"  {_hp_var} (heat pump power) = {_raw_hp!r}")
                print(f"  power_measurement_vars       = {pv}")
                print(f"  zone_temp used: {zone_temp_K:.4f} K = {zone_temp_C:.4f} °C")
                if zone_temp_K == 293.15:
                    print(f"  ⚠  WARNING: zone_temp is the 293.15 K fallback!")
                    print(f"     Check that BOPTEST is running and reaTZon_y is in /measurements/")
                print()

            # External temperature — from FORECAST, not measurements.
            # BOPTEST returns dry-bulb temp in Kelvin (>200); anything ≤200
            # means the forecast fetch failed (default 0.0) — show NaN so the
            # plot clearly indicates missing data rather than a spurious 0 °C.
            outdoor_temp_K = fc_values['TDryBul']
            if outdoor_temp_K > 200:
                outdoor_temp_C = outdoor_temp_K - 273.15
            elif outdoor_temp_K > -50:   # may already be in Celsius
                outdoor_temp_C = outdoor_temp_K
            else:
                outdoor_temp_C = float('nan')

            # Power — read using the already-resolved variable names
            heat_pump_power = extract_value(measurements, pv['heat_pump'], 0) if pv.get('heat_pump') else 0.0
            fan_power       = extract_value(measurements, pv['fan'],       0) if pv.get('fan')       else 0.0
            pump_power      = extract_value(measurements, pv['pump'],      0) if pv.get('pump')      else 0.0
            total_power = heat_pump_power + fan_power + pump_power

            solar_rad  = fc_values['HDirNor']
            elec_price = fc_values['PriceElectricPowerHighlyDynamic']

            # Internal gains — variable name differs by testcase; graceful fallback.
            internal_gains = 0.0
            for ig_var in ('InternalGainsPeo_y', 'InternalGainsRad_y',
                           'InternalGains_y', 'InternalGainsRad'):
                if ig_var in measurements:
                    internal_gains = extract_value(measurements, ig_var, 0)
                    break

            # Setpoints come from forecast, not measurements, and have no [1]
            # suffix for single-zone testcases (bestest_hydronic_heat_pump).
            lower_setp = fc_values.get('LowerSetp', 293.15)   # K
            upper_setp = fc_values.get('UpperSetp', 298.15)   # K

            sim_time = extract_value(measurements, 'time', 0)
            
            # Compute per-step comfort bounds (dynamic schedule or fallback)
            if 'comfort_temp_lower_K' in info:
                step_comfort_low_C  = info['comfort_temp_lower_K'] - 273.15
                step_comfort_high_C = info['comfort_temp_upper_K'] - 273.15
            elif reward_wrapper is not None:
                lo_K, hi_K = reward_wrapper._get_comfort_range(sim_time)
                step_comfort_low_C  = lo_K - 273.15
                step_comfort_high_C = hi_K - 273.15
            else:
                step_comfort_low_C, step_comfort_high_C = comfort_low_C, comfort_high_C

            # Calculate discomfort for this step using dynamic comfort bounds
            if zone_temp_C < step_comfort_low_C:
                step_discomfort = (step_comfort_low_C - zone_temp_C) ** 2
            elif zone_temp_C > step_comfort_high_C:
                step_discomfort = (zone_temp_C - step_comfort_high_C) ** 2
            else:
                step_discomfort = 0
            
            # Accumulate metrics
            cumulative_reward += reward
            cumulative_energy += total_power / 1000  # Convert to kW
            cumulative_discomfort += step_discomfort
            
            # Store data
            data['steps'].append(step)
            data['times'].append(sim_time)
            data['zone_temps'].append(zone_temp_C)
            data['zone_temps_K'].append(zone_temp_K)
            data['outdoor_temps'].append(outdoor_temp_C)
            data['actions'].append(action[0] if hasattr(action, '__len__') else action)
            data['heat_pump_power'].append(heat_pump_power)
            data['fan_power'].append(fan_power)
            data['pump_power'].append(pump_power)
            data['total_power'].append(total_power)
            data['solar_radiation'].append(solar_rad)
            data['internal_gains'].append(internal_gains)
            data['electricity_price'].append(elec_price)
            data['lower_setpoint'].append(lower_setp - 273.15)
            data['upper_setpoint'].append(upper_setp - 273.15)
            data['rewards'].append(reward)
            data['comfort_rewards'].append(info.get('comfort_reward', 0))
            data['price_rewards'].append(info.get('price_reward', 0))
            data['cumulative_reward'].append(cumulative_reward)
            data['cumulative_energy'].append(cumulative_energy)
            data['cumulative_discomfort'].append(cumulative_discomfort)
            data['lower_setpoint_comfort'].append(step_comfort_low_C)
            data['upper_setpoint_comfort'].append(step_comfort_high_C)
            
            # Print every ~3 hours of simulation time regardless of step_period
            _sp = data['step_period']
            _print_every = max(1, int(round(3 * 3600 / _sp)))  # steps per 3 simulated hours
            elapsed_h = step * _sp / 3600.0
            if step % _print_every == 0:
                out_str = f"{outdoor_temp_C:5.1f}°C" if not np.isnan(outdoor_temp_C) else "  N/A "
                c_rew = info.get('comfort_reward', 0)
                p_rew = info.get('price_reward', 0)
                print(f"Step {step:3d} | Time: {elapsed_h:5.1f}h | "
                      f"Zone: {zone_temp_C:5.2f}°C | "
                      f"Out: {out_str} | "
                      f"Action: {data['actions'][-1]:4.2f} | "
                      f"Power: {total_power:6.0f}W | "
                      f"Comf: {c_rew:7.3f} | "
                      f"Price: {p_rew:7.3f} | "
                      f"Total: {reward:7.3f}")
        
        step += 1
    
    # Get final KPIs
    kpis = base_env.get_kpis()
    data['kpis'] = kpis
    
    print(f"\nEpisode {episode_num} Complete!")
    print(f"Total Steps: {step}")
    print(f"Total Reward: {cumulative_reward:.3f}")
    print(f"Final KPIs:")
    print(f"  Energy: {kpis.get('ener_tot', 0):.4f}")
    print(f"  Discomfort: {kpis.get('tdis_tot', 0):.4f}")
    print(f"  Cost: {kpis.get('cost_tot', 0):.4f}")
    
    return data


def plot_episode_detailed(episode_data, episode_num, save_path=None):
    """
    Create comprehensive visualization for a single episode.
    9 subplots across 3 pages:
      Page 1 - Temperature & Comfort
      Page 2 - Actions & Power
      Page 3 - Reward & Diagnostics
    """

    steps  = episode_data['steps']
    n      = len(steps)

    # Convert step indices to elapsed simulation hours using the recorded step_period.
    # With step_period=1800 s (30 min): step 0 = 0.5 h, step 1 = 1.0 h, etc.
    _sp_h  = episode_data.get('step_period', 1800) / 3600.0
    hours  = (np.array(steps) + 1) * _sp_h   # step 0 ends at +1 step_period
    x_max  = float(hours[-1]) if n > 0 else 1.0

    # Read comfort bounds stored in the data dict (matches whatever the model was trained with)
    COMFORT_LOW  = episode_data.get('comfort_low_C',  22.0)
    COMFORT_HIGH = episode_data.get('comfort_high_C', 24.0)
    lo_arr_plot = episode_data.get('lower_setpoint_comfort')
    hi_arr_plot = episode_data.get('upper_setpoint_comfort')
    discomfort_per_step = []
    for idx, t in enumerate(episode_data['zone_temps']):
        lo = lo_arr_plot[idx] if lo_arr_plot else COMFORT_LOW
        hi = hi_arr_plot[idx] if hi_arr_plot else COMFORT_HIGH
        if t < lo:
            discomfort_per_step.append((lo - t) ** 2)
        elif t > hi:
            discomfort_per_step.append((t - hi) ** 2)
        else:
            discomfort_per_step.append(0.0)

    temps_in = [
        t for idx, t in enumerate(episode_data['zone_temps'])
        if (lo_arr_plot[idx] if lo_arr_plot else COMFORT_LOW)
           <= t <=
           (hi_arr_plot[idx] if hi_arr_plot else COMFORT_HIGH)
    ]
    pct_comfort = 100 * len(temps_in) / max(n, 1)

    # ── shared style ────────────────────────────────────────────────────────
    def _style(ax, title, xlabel='Time (hours)', ylabel=None):
        ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
        ax.set_xlabel(xlabel, fontsize=9)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlim(0, x_max)
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.tick_params(labelsize=8)

    # ════════════════════════════════════════════════════════════════════════
    # PAGE 1 — TEMPERATURE & COMFORT
    # ════════════════════════════════════════════════════════════════════════
    fig1, axes1 = plt.subplots(3, 1, figsize=(14, 12))
    fig1.suptitle(f'Episode {episode_num}  ·  Temperature & Comfort',
                  fontsize=14, fontweight='bold')

    # --- 1. Internal Temperature -------------------------------------------
    ax = axes1[0]
    ax.plot(hours, episode_data['zone_temps'], color='#2196F3', linewidth=2,
            label='Zone Temperature')
    lo_arr = np.array(episode_data.get('lower_setpoint_comfort', [COMFORT_LOW] * n))
    hi_arr = np.array(episode_data.get('upper_setpoint_comfort', [COMFORT_HIGH] * n))
    ax.fill_between(hours, lo_arr, hi_arr, alpha=0.12, color='#4CAF50',
                    label='Comfort Zone (dynamic)')
    ax.step(hours, lo_arr, where='post', color='#4CAF50', linestyle='--', linewidth=1.5,
            label='Comfort Lower (dynamic)')
    ax.step(hours, hi_arr, where='post', color='#F44336', linestyle='--', linewidth=1.5,
            label='Comfort Upper (dynamic)')
    zone = np.array(episode_data['zone_temps'])
    ax.fill_between(hours, zone, lo_arr, where=(zone < lo_arr),
                    alpha=0.25, color='#2196F3', label='Too cold')
    ax.fill_between(hours, zone, hi_arr, where=(zone > hi_arr),
                    alpha=0.25, color='#F44336', label='Too hot')
    ax.legend(fontsize=8, loc='best', ncol=2)
    _style(ax, f'Internal Zone Temperature  (in comfort: {pct_comfort:.1f}%)',
           ylabel='Temperature (°C)')

    # --- 2. External Temperature -------------------------------------------
    ax = axes1[1]
    out = np.array(episode_data['outdoor_temps'])
    ax.plot(hours, out, color='#FF9800', linewidth=2, label='Outdoor Temperature')
    out_min = float(np.nanmin(out)) if not np.all(np.isnan(out)) else 0.0
    ax.fill_between(hours, out_min, out, alpha=0.15, color='#FF9800')
    ax.legend(fontsize=8)
    _style(ax, 'External (Outdoor) Temperature', ylabel='Temperature (°C)')

    # --- 3. Comfort (discomfort penalty per step) --------------------------
    ax = axes1[2]
    disc = np.array(discomfort_per_step)
    ax.bar(hours, disc, color=np.where(disc > 0, '#E53935', '#66BB6A'),
           width=0.8, label='Discomfort penalty')
    ax.plot(hours, episode_data['cumulative_discomfort'],
            color='#7B1FA2', linewidth=2, linestyle='-',
            label='Cumulative discomfort')
    ax.set_ylabel('Discomfort  [(°C)²]', fontsize=9)
    ax2r = ax.twinx()
    ax2r.plot(hours, episode_data['cumulative_discomfort'],
              color='#7B1FA2', linewidth=2, alpha=0)   # invisible – just sets scale
    ax2r.set_ylabel('Cumulative', fontsize=9, color='#7B1FA2')
    ax2r.tick_params(axis='y', labelcolor='#7B1FA2', labelsize=8)
    ax.legend(fontsize=8, loc='upper left')
    _style(ax, 'Comfort Penalty per Time Step  (0 = fully comfortable)',
           ylabel='Discomfort  [(°C)²]')

    fig1.tight_layout()
    if save_path:
        p1 = save_path.replace('.png', '_page1_temperature.png')
        fig1.savefig(p1, dpi=300, bbox_inches='tight')
        print(f'  Saved: {p1}')

    # ════════════════════════════════════════════════════════════════════════
    # PAGE 2 — ACTIONS & POWER
    # ════════════════════════════════════════════════════════════════════════
    fig2, axes2 = plt.subplots(3, 1, figsize=(14, 12))
    fig2.suptitle(f'Episode {episode_num}  ·  Control Actions & Power',
                  fontsize=14, fontweight='bold')

    # --- 4. Heat pump control action ---------------------------------------
    ax = axes2[0]
    actions = np.array(episode_data['actions'])
    ax.step(hours, actions, where='post', color='#E53935', linewidth=2,
            label='Heat pump signal')
    ax.fill_between(hours, 0, actions, step='post', alpha=0.25, color='#E53935')
    # Overlay zone temp on secondary axis so action decisions make sense
    ax_r = ax.twinx()
    ax_r.plot(hours, episode_data['zone_temps'], color='#2196F3',
              linewidth=1.5, linestyle=':', alpha=0.7, label='Zone temp (°C)')
    lo_arr = np.array(episode_data.get('lower_setpoint_comfort', [COMFORT_LOW] * n))
    hi_arr = np.array(episode_data.get('upper_setpoint_comfort', [COMFORT_HIGH] * n))
    ax_r.step(hours, lo_arr, where='post', color='#4CAF50', linestyle='--', linewidth=0.8, alpha=0.6)
    ax_r.step(hours, hi_arr, where='post', color='#F44336', linestyle='--', linewidth=0.8, alpha=0.6)
    ax_r.set_ylabel('Zone Temp (°C)', fontsize=9, color='#2196F3')
    ax_r.tick_params(axis='y', labelcolor='#2196F3', labelsize=8)
    ax.set_ylim(-0.05, 1.1)
    ax.set_ylabel('Control Signal [0–1]', fontsize=9)
    # Combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='best')
    _style(ax, 'Heat Pump Control Action vs Zone Temperature')

    # --- 5. Total Power vs Time -------------------------------------------
    ax = axes2[1]
    ax.plot(hours, episode_data['total_power'], color='#212121',
            linewidth=2, label='Total Power')
    ax.plot(hours, episode_data['heat_pump_power'], color='#E53935',
            linewidth=1.5, linestyle='--', label='Heat Pump', alpha=0.8)
    ax.plot(hours, episode_data['fan_power'], color='#1565C0',
            linewidth=1.2, linestyle=':', label='Fan', alpha=0.8)
    ax.plot(hours, episode_data['pump_power'], color='#2E7D32',
            linewidth=1.2, linestyle=':', label='Emission Pump', alpha=0.8)
    ax.fill_between(hours, 0, episode_data['total_power'],
                    alpha=0.12, color='#212121')
    ax.legend(fontsize=8, loc='best')
    _style(ax, 'Total Power Consumption (with Breakdown)', ylabel='Power (W)')

    # --- 6. Electricity price vs action (are they correlated?) ------------
    ax = axes2[2]
    if n > 0:
        price = np.array(episode_data['electricity_price'])
        # Normalise price for visual overlap
        p_min, p_max = float(price.min()), float(price.max())
        price_norm = (price - p_min) / (p_max - p_min + 1e-9)
        ax.plot(hours, price_norm, color='#FFA000', linewidth=2,
                label='Electricity Price (normalised)')
        ax.fill_between(hours, 0, price_norm, alpha=0.15, color='#FFA000')
        ax.step(hours, actions, where='post', color='#E53935', linewidth=1.5,
                linestyle='--', label='Heat Pump Action', alpha=0.8)
        ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=8, loc='best')
    _style(ax, 'Electricity Price vs Heat Pump Action  (price-responsive check)',
           ylabel='Normalised Value')

    fig2.tight_layout()
    if save_path:
        p2 = save_path.replace('.png', '_page2_actions_power.png')
        fig2.savefig(p2, dpi=300, bbox_inches='tight')
        print(f'  Saved: {p2}')

    # ════════════════════════════════════════════════════════════════════════
    # PAGE 3 — REWARD & DIAGNOSTICS
    # ════════════════════════════════════════════════════════════════════════
    fig3, axes3 = plt.subplots(3, 1, figsize=(14, 12))
    fig3.suptitle(f'Episode {episode_num}  ·  Reward & Diagnostics',
                  fontsize=14, fontweight='bold')

    # --- 7. Per-step reward breakdown -------------------------------------
    ax = axes3[0]
    cr = np.array(episode_data['comfort_rewards'])
    er = np.array(episode_data['price_rewards'])
    tr = np.array(episode_data['rewards'])
    ax.bar(hours, cr, width=0.8, color='#43A047', alpha=0.7, label='Comfort reward')
    ax.bar(hours, er, width=0.8, color='#E53935', alpha=0.7,
           bottom=cr, label='Price reward')
    ax.plot(hours, tr, color='#1A237E', linewidth=2, label='Total reward')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.legend(fontsize=8, loc='best')
    _style(ax, 'Per-Step Reward (stacked: comfort + energy)',
           ylabel='Reward')

    # --- 8. Cumulative reward over time -----------------------------------
    ax = axes3[1]
    cum_r = np.array(episode_data['cumulative_reward'])
    ax.plot(hours, cum_r, color='#1A237E', linewidth=2.5,
            label='Cumulative Reward')
    if n > 0:
        cum_r_min = float(np.nanmin(cum_r)) if len(cum_r) > 0 else 0.0
        ax.fill_between(hours, cum_r_min, cum_r,
                        where=(cum_r >= cum_r[0]), alpha=0.15, color='#1A237E')
        # Annotate final value
        ax.annotate(f'Final: {cum_r[-1]:.2f}',
                    xy=(hours[-1], cum_r[-1]),
                    xytext=(-40, 12), textcoords='offset points',
                    fontsize=9, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#1A237E'))
    ax.legend(fontsize=8)
    _style(ax, 'Cumulative Reward Accumulated Over Episode',
           ylabel='Cumulative Reward')

    # --- 9. Action vs Zone Temp scatter (what did model learn?) -----------
    ax = axes3[2]
    if n > 0:
        sc = ax.scatter(episode_data['zone_temps'], actions,
                        c=hours, cmap='viridis', s=30, alpha=0.7,
                        label='Steps (colour = time)')
        plt.colorbar(sc, ax=ax, label='Time (hours)')
    ax.axvline(COMFORT_LOW,  color='#4CAF50', linestyle='--', linewidth=1.2,
               label=f'Comfort lower ({COMFORT_LOW}°C)')
    ax.axvline(COMFORT_HIGH, color='#F44336', linestyle='--', linewidth=1.2,
               label=f'Comfort upper ({COMFORT_HIGH}°C)')
    ax.set_xlabel('Zone Temperature (°C)', fontsize=9)
    ax.set_ylabel('Heat Pump Action [0–1]', fontsize=9)
    ax.set_title('Policy Behaviour: Action vs Zone Temperature\n'
                 '(colour = time through episode)',
                 fontsize=11, fontweight='bold', pad=6)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.tick_params(labelsize=8)

    fig3.tight_layout()
    if save_path:
        p3 = save_path.replace('.png', '_page3_reward_diagnostics.png')
        fig3.savefig(p3, dpi=300, bbox_inches='tight')
        print(f'  Saved: {p3}')

    plt.show()


def visualize_trained_model(model_dir, n_episodes=2, url='http://127.0.0.1:8000'):
    """
    Load a trained model and visualize its performance over multiple episodes.
    
    Parameters
    ----------
    model_dir : str
        Directory containing the trained model
    n_episodes : int
        Number of episodes to run and visualize
    url : str
        BOPTEST server URL
    """
    
    _require_sb3()

    print("="*80)
    print("TRAINED MODEL EPISODE VISUALIZATION")
    print("="*80)
    print(f"Model directory: {model_dir}")
    print(f"Number of episodes: {n_episodes}")
    print()
    
    # Load configuration
    config_path = os.path.join(model_dir, 'config.json')
    if not os.path.exists(config_path):
        print(f"Error: config.json not found in {model_dir}")
        return
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("Model Configuration:")
    print(f"  Algorithm:          {config.get('algorithm', 'Unknown')}")
    print(f"  Comfort Weight:     {config.get('comfort_weight', 0.5)}")
    print(f"  Price Weight:       {config.get('price_weight', 100.0)}")
    print(f"  Training Timesteps: {config.get('total_timesteps', 'Unknown')}")
    print(f"  Observation dim:    {config.get('observation_dim', 'Unknown')}")

    trained_steps = config.get('total_timesteps', 0)
    trained_algo  = config.get('algorithm', 'SAC')
    if isinstance(trained_steps, int) and 0 < trained_steps < 20000:
        print(f"\n⚠  WARNING: Model was trained with only {trained_steps:,} timesteps.")
        print(f"   {trained_algo} needs substantially more experience to learn a useful policy.")
        print(f"   A minimum of 50,000 timesteps is recommended (200,000+ preferred).")
        print(f"   Near-zero actions, zero power, and constant zone temperature are")
        print(f"   expected symptoms of an undertrained model.")
        print(f"   Retrain with total_timesteps=200000 (or more) for better results.\n")
    
    env_params = config.get('env_params', None)
    if env_params:
        def _h(val):
            """Convert seconds to hours string, or '?' if not a number."""
            return f"{int(val) // 3600}" if isinstance(val, (int, float)) else '?'
        print(f"  Episode length:     {_h(env_params.get('max_episode_length'))} hours")
        print(f"  Predictive period:  {_h(env_params.get('predictive_period'))} hours")
        print(f"  Regressive period:  {_h(env_params.get('regressive_period', 0))} hours")
        cr = env_params.get('comfort_temp_range', [295.15, 297.15])
        print(f"  Comfort range:      {cr[0]-273.15:.0f}–{cr[1]-273.15:.0f}°C")
    else:
        print("  WARNING: env_params not found in config — using current defaults.")
        print("           If observation dim mismatches the model will crash.")
    
    # Load model
    algorithm = config.get('algorithm', 'PPO')
    model_path = os.path.join(model_dir, 'final_model.zip')
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    print(f"\nLoading {algorithm} model...")
    if algorithm == 'PPO':
        model = PPO.load(model_path)
    elif algorithm == 'DQN':
        model = DQN.load(model_path)
    elif algorithm == 'SAC':
        model = SAC.load(model_path)
    else:
        print(f"Unknown algorithm: {algorithm}")
        return
    
    print("Model loaded successfully!")
    
    # Re-create the environment with the EXACT same params used during training,
    # but override max_episode_length so each visualisation episode runs for 300 steps.
    VIS_STEPS = 300
    vis_env_params = dict(env_params) if env_params is not None else {}
    step_period = vis_env_params.get('step_period', 1800)
    vis_env_params['max_episode_length'] = VIS_STEPS * step_period

    print(f"\nCreating environment (matching training config, {VIS_STEPS} steps per episode)...")
    env = create_env(
        url=url,
        testcase=config.get('testcase', 'bestest_hydronic_heat_pump'),
        use_custom_reward=True,
        comfort_weight=config.get('comfort_weight', 0.5),
        price_weight=config.get('price_weight', 100.0),
        use_discrete_actions=(algorithm == 'DQN'),
        env_params=vis_env_params,       # ← restores training params + 300-step episode length
    )
    
    actual_dim = env.observation_space.shape[0]
    expected_dim = config.get('observation_dim', actual_dim)
    if actual_dim != expected_dim:
        print(f"\nERROR: Observation space mismatch!")
        print(f"  Model expects: ({expected_dim},)")
        print(f"  Env produces:  ({actual_dim},)")
        print("  env_params in config.json may be incomplete or missing.")
        env.close()
        return
    
    print(f"Environment created! Observation space: {env.observation_space.shape}")
    
    # ── Deterministic, distinct start times per episode ─────────────────────
    # Both windows are in January (mid-winter) so conditions are comparable,
    # but they are 7 days apart so the agent sees a genuinely different day.
    #   Episode 1 → Jan 14 00:00  (13 × 86 400 = 1 123 200 s from year start)
    #   Episode 2 → Jan 21 00:00  (20 × 86 400 = 1 728 000 s from year start)
    # Any additional episodes cycle through subsequent Janurary weeks.
    _JAN_START_SEC = [i * 7 * 86400 + 13 * 86400 for i in range(max(n_episodes, 2))]
    #  i=0 → 13 days from Jan 1 = Jan 14
    #  i=1 → 20 days from Jan 1 = Jan 21
    #  i=2 → 27 days from Jan 1 = Jan 28  … still January / early winter

    # Resolve base_env once so we can patch start_time before each reset
    base_env_vis = env
    while hasattr(base_env_vis, 'env'):
        base_env_vis = base_env_vis.env

    # Run episodes and visualize
    all_episode_data = []

    for ep in range(n_episodes):
        # Pin a distinct same-season start time for each episode
        ep_start = _JAN_START_SEC[ep]
        base_env_vis.start_time = ep_start
        base_env_vis.random_start_time = False   # ensure reset() uses our value

        _day_of_year = ep_start // 86400 + 1     # 1-indexed calendar day
        print(f'\nEpisode {ep+1} start time: {ep_start:,} s  '
              f'(Jan {_day_of_year}, day {_day_of_year} of year)')

        episode_data = run_episode_with_logging(env, model, episode_num=ep+1)
        all_episode_data.append(episode_data)

        # Base path — plot_episode_detailed appends _page1/2/3 suffixes
        save_path = os.path.join(model_dir, f'episode_{ep+1}.png')
        print(f'\nGenerating plots for episode {ep+1}...')
        plot_episode_detailed(episode_data, episode_num=ep+1, save_path=save_path)
    
    # Create comparison plot if multiple episodes
    if n_episodes > 1:
        create_episode_comparison(all_episode_data, model_dir)
    
    env.close()
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"Generated plots in: {model_dir}")
    print("Files per episode:")
    print("  episode_N_page1_temperature.png     — Internal & external temp, comfort")
    print("  episode_N_page2_actions_power.png   — Actions, power, price-response")
    print("  episode_N_page3_reward_diagnostics.png — Reward, cumulative, policy scatter")
    if n_episodes > 1:
        print("  episodes_comparison.png             — Cross-episode comparison")


def create_episode_comparison(all_episode_data, save_dir):
    """Create comparison plot across multiple episodes."""
    
    n_episodes = len(all_episode_data)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Performance Comparison Across {n_episodes} Episodes', 
                 fontsize=16, fontweight='bold')
    
    colors = plt.cm.viridis(np.linspace(0, 1, n_episodes))

    def _ep_hours(data):
        """Convert episode step indices to elapsed hours using step_period."""
        steps = data['steps']
        if not steps:
            return np.array([], dtype=float)
        sp_h = data.get('step_period', 1800) / 3600.0
        return (np.array(steps) + 1) * sp_h

    # Comfort bounds from episode data (not hardcoded 22/24)
    _ep0 = all_episode_data[0]
    _comfort_lo = _ep0.get('comfort_low_C',  22.0)
    _comfort_hi = _ep0.get('comfort_high_C', 24.0)

    # Temperature comparison
    for i, data in enumerate(all_episode_data):
        ep_h = _ep_hours(data)
        axes[0, 0].plot(ep_h, data['zone_temps'],
                       color=colors[i], linewidth=2, label=f'Episode {i+1}')
    axes[0, 0].axhline(_comfort_lo, color='g', linestyle='--', alpha=0.5,
                       label=f'Lower ({_comfort_lo:.0f}°C)')
    axes[0, 0].axhline(_comfort_hi, color='r', linestyle='--', alpha=0.5,
                       label=f'Upper ({_comfort_hi:.0f}°C)')
    _ep0_h = _ep_hours(_ep0)
    _x_max = float(_ep0_h[-1]) if len(_ep0_h) > 0 else 1.0
    axes[0, 0].fill_between([0, _x_max], _comfort_lo, _comfort_hi,
                            alpha=0.1, color='green')
    axes[0, 0].set_ylabel('Temperature (°C)', fontweight='bold')
    axes[0, 0].set_xlabel('Time (hours)')
    axes[0, 0].set_title('Zone Temperature')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    # Actions comparison
    for i, data in enumerate(all_episode_data):
        ep_h = _ep_hours(data)
        axes[0, 1].plot(ep_h, data['actions'],
                       color=colors[i], linewidth=2, label=f'Episode {i+1}')
    axes[0, 1].set_ylabel('Control Signal', fontweight='bold')
    axes[0, 1].set_xlabel('Time (hours)')
    axes[0, 1].set_title('Heat Pump Control')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Power comparison
    for i, data in enumerate(all_episode_data):
        ep_h = _ep_hours(data)
        axes[1, 0].plot(ep_h, data['total_power'],
                       color=colors[i], linewidth=2, label=f'Episode {i+1}')
    axes[1, 0].set_ylabel('Power (W)', fontweight='bold')
    axes[1, 0].set_xlabel('Time (hours)')
    axes[1, 0].set_title('Total Power Consumption')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Summary bar chart
    metrics = ['Energy', 'Discomfort', 'Cost']
    x = np.arange(len(metrics))
    width = 0.8 / n_episodes
    
    for i, data in enumerate(all_episode_data):
        kpis = data['kpis']
        values = [
            kpis.get('ener_tot', 0),
            kpis.get('tdis_tot', 0),
            kpis.get('cost_tot', 0)
        ]
        axes[1, 1].bar(x + i*width, values, width, 
                      label=f'Episode {i+1}', color=colors[i])
    
    axes[1, 1].set_ylabel('KPI Value', fontweight='bold')
    axes[1, 1].set_title('KPI Comparison')
    axes[1, 1].set_xticks(x + width * (n_episodes-1) / 2)
    axes[1, 1].set_xticklabels(metrics)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'episodes_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved comparison plot to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Visualize trained BOPTEST RL model performance over episodes'
    )
    parser.add_argument('model_dir', type=str, 
                       help='Directory containing trained model')
    parser.add_argument('--episodes', type=int, default=2,
                       help='Number of episodes to run (default: 2)')
    parser.add_argument('--url', type=str, default='http://127.0.0.1:8000',
                       help='BOPTEST server URL (default: http://127.0.0.1:8000)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_dir):
        print(f"Error: Directory not found: {args.model_dir}")
        print("\nAvailable models:")
        if os.path.exists('trained_models'):
            for subdir in sorted(os.listdir('trained_models')):
                print(f"  trained_models/{subdir}")
        exit(1)
    
    visualize_trained_model(args.model_dir, n_episodes=args.episodes, url=args.url)
