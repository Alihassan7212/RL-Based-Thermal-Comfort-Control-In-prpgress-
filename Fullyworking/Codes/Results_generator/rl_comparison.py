"""
rl_comparison.py — Side-by-side comparison of up to 3 trained RL models
========================================================================

Runs N episodes of each supplied RL model against the same BOPTEST testcase
with identical episode start times, so results are directly comparable.

Output
------
  <save_dir>/
    episode_<N>_comparison.png   — 4-panel overlay (zone temp, signal, power, outdoor)
    episode_<N>_kpi_bars.png     — grouped KPI bar chart
    multi_episode_summary.png    — mean ± std across all episodes (if N > 1)
    summary.txt                  — full results table
    summary.csv                  — machine-readable results

Usage
-----
    python rl_comparison.py <model_dir_1> <model_dir_2> <model_dir_3> \\
        [--names "SAC-v1,SAC-v2,SAC-v3"] \\
        [--episodes 2] \\
        [--url http://127.0.0.1:8000] \\
        [--save-dir /path/to/output]

Notes
-----
- Each model directory must contain final_model.zip and config.json.
- Models may have different observation spaces (backward compatibility is
  handled automatically via config.json env_params).
- The script uses one independent BOPTEST environment per model so that
  the simulations do not interfere with each other.
"""

import sys
import os
import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from stable_baselines3 import PPO, DQN, SAC
except ModuleNotFoundError:
    PPO = DQN = SAC = None

from train_rl import create_env


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_SCHEDULE = [
    (7,  22, (294.15, 297.15)),   # day   07:00–22:00 → 21–24 °C
    (22,  7, (293.15, 295.15)),   # night 22:00–07:00 → 20–22 °C
]

# Per-model colour + line style (up to 3 models)
_MODEL_STYLES = [
    {'color': '#2196F3', 'ls': '-',  'lw': 2.0},   # blue  solid
    {'color': '#E53935', 'ls': '--', 'lw': 1.8},   # red   dashed
    {'color': '#9C27B0', 'ls': '-.', 'lw': 1.8},   # purple dash-dot
]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _require_sb3():
    if any(v is None for v in (PPO, DQN, SAC)):
        raise RuntimeError(
            "stable_baselines3 is required. Install: pip install stable-baselines3"
        )


def extract_value(data, key, default=0.0):
    """Extract a numeric value from a BOPTEST measurement payload entry."""
    val = data.get(key, default)
    if isinstance(val, dict):
        v = val.get('value', default)
        return float(default if v is None else v)
    return float(default if val is None else val)


def _get_comfort_range_from_schedule(schedule, sim_time_s):
    """Return (lower_K, upper_K) from a [(start_h, end_h, (lo, hi)), ...] schedule."""
    hour = (sim_time_s % 86400.0) / 3600.0
    for start_h, end_h, bounds in schedule:
        lo, hi = float(bounds[0]), float(bounds[1])
        if start_h < end_h:
            if start_h <= hour < end_h:
                return lo, hi
        else:
            if hour >= start_h or hour < end_h:
                return lo, hi
    return float(bounds[0]), float(bounds[1])


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
    ax.tick_params(labelsize=8, labelbottom=True)


# ──────────────────────────────────────────────────────────────────────────────
# RL episode runner
# ──────────────────────────────────────────────────────────────────────────────

def run_rl_episode(env, model, base_env, episode_num: int,
                   comfort_schedule, step_period: int,
                   model_label: str) -> dict:
    """
    Run one episode of ``model`` inside ``env`` and return a data dict.

    Parameters
    ----------
    env           : fully-wrapped Gymnasium env (output of create_env)
    model         : loaded SB3 model (SAC / PPO / DQN)
    base_env      : unwrapped BoptestGymEnv (for direct cache access)
    episode_num   : 1-based episode index (for logging)
    comfort_schedule : [(start_h, end_h, (lo_K, hi_K)), ...]
    step_period   : simulation step length in seconds
    model_label   : display name for this model
    """
    step_h = step_period / 3600.0

    # Find reward wrapper for comfort bound lookup
    reward_wrapper = None
    _tmp = env
    while hasattr(_tmp, 'env'):
        if hasattr(_tmp, '_get_comfort_range'):
            reward_wrapper = _tmp
            break
        _tmp = _tmp.env

    obs, _ = env.reset()

    # Seed energy tracking from reset KPIs
    prev_kpi_energy = float(getattr(base_env, 'last_kpis', {}).get('ener_tot', 0.0))

    data = {
        'model':    model_label,
        'episode':  episode_num,
        'steps':    [], 'times': [],
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

    cum_reward   = 0.0
    cum_disc_Kh  = 0.0
    n_in_band    = 0
    step         = 0
    done         = False

    print(f"\n{'─'*60}")
    print(f"  {model_label}  |  Episode {episode_num}")
    print(f"{'─'*60}")

    while not done:
        action_arr, _ = model.predict(obs, deterministic=True)
        action_val = float(action_arr[0] if hasattr(action_arr, '__len__') else action_arr)

        obs, reward, terminated, truncated, info = env.step(action_arr)
        done = terminated or truncated

        # Read measurements from env cache (no extra HTTP calls)
        _res  = getattr(base_env, 'res', None) or {}
        meas  = getattr(base_env, 'last_measurements', None) or _res

        zone_K = extract_value(meas, 'reaTZon_y', 293.15)
        if zone_K == 293.15 and _res:
            zone_K = extract_value(_res, 'reaTZon_y', 293.15)
        zone_C = zone_K - 273.15
        sim_t  = extract_value(meas, 'time', 0.0)
        if sim_t == 0.0 and _res:
            sim_t = extract_value(_res, 'time', 0.0)

        # Outdoor temperature from forecast cache
        _last_fc = getattr(base_env, 'last_forecast', {})
        out_K = float(_last_fc.get('TDryBul', [0.0])[0]) if isinstance(
            _last_fc.get('TDryBul'), list) and _last_fc.get('TDryBul') else float(
            _last_fc.get('TDryBul', 0.0) if not isinstance(
                _last_fc.get('TDryBul'), list) else 0.0)
        out_C = (out_K - 273.15) if out_K > 200 else float('nan')

        # Comfort bounds
        if 'comfort_temp_lower_K' in info:
            lo_K = info['comfort_temp_lower_K']
            hi_K = info['comfort_temp_upper_K']
        elif reward_wrapper is not None:
            lo_K, hi_K = reward_wrapper._get_comfort_range(sim_t)
        else:
            lo_K, hi_K = _get_comfort_range_from_schedule(comfort_schedule, sim_t)

        # Power via KPI-delta method (W/m²), matching PBRSRewardWrapper exactly
        _kpis_now = getattr(base_env, 'last_kpis', {})
        _curr_en  = float(_kpis_now.get('ener_tot', prev_kpi_energy))
        _delta_en = max(0.0, _curr_en - prev_kpi_energy)
        tot_pw    = (_delta_en * 1000.0) / step_h if step_h > 0 else 0.0
        prev_kpi_energy = _curr_en

        # Best-effort measurement-based power breakdown
        pv = getattr(base_env, 'power_measurement_vars', {})
        hp_pw  = extract_value(meas, pv.get('heat_pump', ''), 0.0) if pv.get('heat_pump') else 0.0
        fan_pw = extract_value(meas, pv.get('fan',       ''), 0.0) if pv.get('fan')       else 0.0
        pmp_pw = extract_value(meas, pv.get('pump',      ''), 0.0) if pv.get('pump')      else 0.0
        _meas_total = hp_pw + fan_pw + pmp_pw
        if _meas_total > 0.0:
            tot_pw = _meas_total
        else:
            hp_pw = tot_pw

        # Discomfort
        e_c = max(0.0, lo_K - zone_K, zone_K - hi_K)
        disc_Kh = e_c * step_h
        cum_disc_Kh += disc_Kh
        if e_c == 0.0:
            n_in_band += 1

        cum_reward += reward
        c_rew = info.get('comfort_reward', 0.0)
        e_rew = info.get('price_reward',   0.0)

        data['steps'].append(step)
        data['times'].append(sim_t)
        data['zone_temps_C'].append(zone_C)
        data['zone_temps_K'].append(zone_K)
        data['outdoor_temps_C'].append(out_C)
        data['actions'].append(action_val)
        data['heat_pump_power'].append(hp_pw)
        data['fan_power'].append(fan_pw)
        data['pump_power'].append(pmp_pw)
        data['total_power'].append(tot_pw)
        data['lower_setpoint_C'].append(lo_K - 273.15)
        data['upper_setpoint_C'].append(hi_K - 273.15)
        data['rewards'].append(reward)
        data['comfort_rewards'].append(c_rew)
        data['energy_rewards'].append(e_rew)
        data['cumulative_reward'].append(cum_reward)
        data['cumulative_discomfort_Kh'].append(cum_disc_Kh)

        _print_every = max(1, int(round(6 * 3600 / step_period)))
        if step % _print_every == 0:
            elapsed_h = step * step_h
            cf = "✓" if e_c == 0.0 else f"✗ ({e_c:.2f} K)"
            print(f"  step {step:3d} | {elapsed_h:5.1f}h | "
                  f"zone {zone_C:5.2f}°C [{lo_K-273.15:.1f}–{hi_K-273.15:.1f}] {cf} | "
                  f"act {action_val:.2f} | pwr {tot_pw:6.2f} W/m²")
        step += 1

    kpis = base_env.get_kpis()
    data['kpis'] = {k: (float(v) if v is not None else 0.0) for k, v in kpis.items()}
    data['pct_time_comfortable'] = 100.0 * n_in_band / max(len(data['steps']), 1)
    data['total_steps'] = len(data['steps'])
    data['cumulative_reward_final'] = cum_reward

    print(f"\n  ── {model_label} Episode {episode_num} complete ({len(data['steps'])} steps) ──")
    print(f"  Energy (ener_tot):     {data['kpis'].get('ener_tot', 0):.4f}  kWh/m²")
    print(f"  Discomfort (tdis_tot): {data['kpis'].get('tdis_tot', 0):.4f}  K·h/m²")
    print(f"  Cost (cost_tot):       {data['kpis'].get('cost_tot', 0):.4f}  EUR/m²")
    print(f"  In-band:               {data['pct_time_comfortable']:.1f}%")
    print(f"  Cumulative reward:     {cum_reward:.2f}")

    return data


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_combined_episode(all_data: dict, model_names: list, save_dir: str,
                          episode_num: int = 1):
    """
    One figure per episode with all models overlaid on shared subplots.

    Produces:
      episode_<N>_comparison.png  — 4-panel overlay
      episode_<N>_kpi_bars.png    — grouped KPI bar chart
    """
    idx  = episode_num - 1
    names_present = [n for n in model_names
                     if n in all_data and idx < len(all_data[n])]
    if not names_present:
        return

    datasets = {n: all_data[n][idx] for n in names_present}
    styles   = {n: _MODEL_STYLES[i % len(_MODEL_STYLES)]
                for i, n in enumerate(model_names) if n in names_present}

    ref_data = datasets[names_present[0]]
    hrs_ref  = _hours(ref_data)
    x_max    = float(hrs_ref[-1]) if len(hrs_ref) > 0 else 1.0
    lo_arr   = np.array(ref_data['lower_setpoint_C'])
    hi_arr   = np.array(ref_data['upper_setpoint_C'])

    fig, axes = plt.subplots(4, 1, figsize=(15, 16), sharex=True)
    fig.suptitle(f'RL Model Comparison — Episode {episode_num}',
                 fontsize=14, fontweight='bold', y=0.998)

    # ── Panel 1: Zone Temperature ──────────────────────────────────────────
    ax = axes[0]
    ax.fill_between(hrs_ref, lo_arr, hi_arr, alpha=0.08, color='#4CAF50',
                    label='Comfort band')
    ax.step(hrs_ref, lo_arr, where='post', color='#4CAF50', ls='--', lw=1.0,
            alpha=0.55, label='Lower setpoint')
    ax.step(hrs_ref, hi_arr, where='post', color='#F44336', ls='--', lw=1.0,
            alpha=0.55, label='Upper setpoint')
    for n in names_present:
        st   = styles[n]
        hrs  = _hours(datasets[n])
        zone = np.array(datasets[n]['zone_temps_C'])
        ax.plot(hrs, zone, color=st['color'], ls=st['ls'], lw=st['lw'], label=n)
    ax.legend(fontsize=9, loc='best', framealpha=0.85)
    _style(ax, 'Zone Temperature', ylabel='Temperature (°C)', x_max=x_max)

    # ── Panel 2: Control Signal ─────────────────────────────────────────────
    ax = axes[1]
    for n in names_present:
        st   = styles[n]
        hrs  = _hours(datasets[n])
        acts = np.array(datasets[n]['actions'])
        ax.step(hrs, acts, where='post', color=st['color'], ls=st['ls'],
                lw=st['lw'], label=n)
    ax.set_ylim(-0.05, 1.12)
    ax.legend(fontsize=9, loc='best', framealpha=0.85)
    _style(ax, 'Heat Pump Control Signal', ylabel='Signal [0–1]', x_max=x_max)

    # ── Panel 3: Total Power ────────────────────────────────────────────────
    ax = axes[2]
    for n in names_present:
        st  = styles[n]
        hrs = _hours(datasets[n])
        pwr = np.array(datasets[n]['total_power'])
        ax.plot(hrs, pwr, color=st['color'], ls=st['ls'], lw=st['lw'], label=n)
    ax.legend(fontsize=9, loc='best', framealpha=0.85)
    _style(ax, 'Total Power Consumption', ylabel='Power (W/m²)', x_max=x_max)

    # ── Panel 4: Outdoor Temperature ────────────────────────────────────────
    ax = axes[3]
    out = np.array(ref_data.get('outdoor_temps_C', []))
    if len(out) > 0 and not np.all(np.isnan(out)):
        ax.plot(hrs_ref[:len(out)], out, color='#795548', lw=1.8,
                label='Outdoor temperature')
        ax.fill_between(hrs_ref[:len(out)], np.nanmin(out), out,
                        alpha=0.10, color='#795548')
    ax.legend(fontsize=9, loc='best', framealpha=0.85)
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
    n_models = len(names_present)
    x        = np.arange(len(kpi_keys))
    width    = 0.22

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    fig2.suptitle(f'RL Model KPI Comparison — Episode {episode_num}',
                  fontsize=13, fontweight='bold')
    for i, n in enumerate(names_present):
        vals = [datasets[n]['kpis'].get(kk, 0.0) for kk in kpi_keys]
        bars = ax2.bar(x + i * width, vals, width, label=n,
                       color=styles[n]['color'], alpha=0.85)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + max(vals) * 0.01,
                         f'{v:.4f}', ha='center', va='bottom',
                         fontsize=7.5, fontweight='bold')
    ax2.set_xticks(x + width * (n_models - 1) / 2)
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


def plot_multi_episode_summary(all_data: dict, model_names: list, save_dir: str):
    """
    4-panel mean ± std summary across ALL episodes for each model.
    Only produced when n_episodes > 1.
    """
    names_present = [n for n in model_names if n in all_data and all_data[n]]
    styles = {n: _MODEL_STYLES[i % len(_MODEL_STYLES)]
              for i, n in enumerate(model_names) if n in names_present}

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('RL Model Multi-Episode Summary', fontsize=14, fontweight='bold')

    min_len = None
    ref_hrs = None

    for n in names_present:
        episodes = all_data[n]
        col = styles[n]['color']

        _min = min(len(d['zone_temps_C']) for d in episodes)
        if min_len is None or _min < min_len:
            min_len = _min
        ref_hrs = _hours(episodes[0])[:min_len]

        zone_mat = np.array([d['zone_temps_C'][:min_len] for d in episodes])
        acts_mat = np.array([d['actions'][:min_len]      for d in episodes])
        pwr_mat  = np.array([d['total_power'][:min_len]  for d in episodes])

        zm, zs = zone_mat.mean(0), zone_mat.std(0)
        am, as_ = acts_mat.mean(0), acts_mat.std(0)
        pm, ps  = pwr_mat.mean(0),  pwr_mat.std(0)

        label = f'{n} (n={len(episodes)})'
        axes[0, 0].plot(ref_hrs, zm, color=col, lw=2, label=label)
        axes[0, 0].fill_between(ref_hrs, zm-zs, zm+zs, alpha=0.15, color=col)
        axes[0, 1].plot(ref_hrs, am, color=col, lw=2, label=label)
        axes[0, 1].fill_between(ref_hrs, am-as_, am+as_, alpha=0.15, color=col)
        axes[1, 0].plot(ref_hrs, pm, color=col, lw=2, label=label)
        axes[1, 0].fill_between(ref_hrs, pm-ps, pm+ps, alpha=0.15, color=col)

    # Comfort band (from first episode of first model)
    _ref = all_data[names_present[0]][0]
    lo_arr = np.array(_ref['lower_setpoint_C'][:min_len])
    hi_arr = np.array(_ref['upper_setpoint_C'][:min_len])
    axes[0, 0].fill_between(ref_hrs, lo_arr, hi_arr, alpha=0.07, color='#4CAF50')
    axes[0, 0].step(ref_hrs, lo_arr, where='post', color='#4CAF50', ls='--', lw=1, alpha=0.5)
    axes[0, 0].step(ref_hrs, hi_arr, where='post', color='#F44336', ls='--', lw=1, alpha=0.5)

    for ax, title, ylabel in [
        (axes[0, 0], 'Zone Temperature (mean ± std)', '°C'),
        (axes[0, 1], 'Heat Pump Signal (mean ± std)', 'Signal [0–1]'),
        (axes[1, 0], 'Total Power (mean ± std)', 'Power (W/m²)'),
    ]:
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Time (hours)', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.tick_params(labelsize=8, labelbottom=True)

    # KPI bar chart (mean ± std across episodes)
    ax = axes[1, 1]
    kpi_keys  = ['ener_tot', 'tdis_tot', 'cost_tot', 'emis_tot']
    kpi_short = ['Energy', 'Discomfort', 'Cost', 'Emissions']
    x_pos  = np.arange(len(kpi_keys))
    w      = 0.22
    for i, n in enumerate(names_present):
        means = [np.mean([d['kpis'].get(k, 0) for d in all_data[n]]) for k in kpi_keys]
        stds  = [np.std( [d['kpis'].get(k, 0) for d in all_data[n]]) for k in kpi_keys]
        ax.bar(x_pos + i*w, means, w, label=n, color=styles[n]['color'],
               alpha=0.82, yerr=stds, capsize=3, error_kw={'linewidth': 1})
    ax.set_xticks(x_pos + w * (len(names_present)-1) / 2)
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

def compute_summary(all_data: dict, model_names: list) -> list:
    """Aggregate per-model, per-episode results into a flat list of row dicts."""
    rows = []
    for name in model_names:
        episodes = all_data.get(name, [])
        if not episodes:
            continue
        ep_rows = []
        for d in episodes:
            kpis = d['kpis']
            acts = np.array(d['actions'])
            pwr  = np.array(d['total_power'])
            zone = np.array(d['zone_temps_C'])
            row = {
                'Model':           name,
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
                'mean_power':      float(np.mean(pwr)),
                'peak_power':      float(np.max(pwr)) if len(pwr) > 0 else 0.0,
                'mean_zone_C':     float(np.mean(zone)),
                'std_zone_C':      float(np.std(zone)),
                'total_reward':    d.get('cumulative_reward_final', 0.0),
            }
            ep_rows.append(row)
            rows.append(row)

        n = len(ep_rows)
        mean_row = {
            'Model':           name,
            'Episode':         'MEAN',
            'Steps':           int(np.mean([r['Steps'] for r in ep_rows])),
            'ener_tot':        np.mean([r['ener_tot']        for r in ep_rows]),
            'tdis_tot':        np.mean([r['tdis_tot']        for r in ep_rows]),
            'cost_tot':        np.mean([r['cost_tot']        for r in ep_rows]),
            'emis_tot':        np.mean([r['emis_tot']        for r in ep_rows]),
            'pct_comfortable': np.mean([r['pct_comfortable'] for r in ep_rows]),
            'mean_action':     np.mean([r['mean_action']     for r in ep_rows]),
            'std_action':      np.mean([r['std_action']      for r in ep_rows]),
            'on_fraction':     np.mean([r['on_fraction']     for r in ep_rows]),
            'mean_power':      np.mean([r['mean_power']      for r in ep_rows]),
            'peak_power':      np.max( [r['peak_power']      for r in ep_rows]),
            'mean_zone_C':     np.mean([r['mean_zone_C']     for r in ep_rows]),
            'std_zone_C':      np.mean([r['std_zone_C']      for r in ep_rows]),
            'total_reward':    np.mean([r['total_reward']    for r in ep_rows]),
        }
        rows.append(mean_row)

    return rows


def print_summary_table(rows: list):
    SEP = '─' * 125

    print('\n' + '=' * 125)
    print(' RL MODEL COMPARISON  —  BOPTEST KPIs (ground truth, lower is better for all KPIs)')
    print('=' * 125)

    print(f'\n{"Model":<18} {"Ep":>4} {"Steps":>6} '
          f'{"ener_tot":>10} {"tdis_tot":>10} {"cost_tot":>10} {"emis_tot":>10} '
          f'{"In-band%":>9} {"MeanAct":>8} {"OnFrac":>7} '
          f'{"MeanPwr":>9} {"CumReward":>11}')
    print(SEP)

    last_model = None
    for r in rows:
        model = r['Model']
        if model != last_model and last_model is not None:
            print()
        last_model = model

        ep     = str(r['Episode'])
        bold   = (ep == 'MEAN')
        prefix = '► ' if bold else '  '

        print(f"{prefix}{model:<16} {ep:>4} {r['Steps']:>6} "
              f"{r['ener_tot']:>10.4f} {r['tdis_tot']:>10.4f} {r['cost_tot']:>10.4f} "
              f"{r['emis_tot']:>10.4f} {r['pct_comfortable']:>9.1f} "
              f"{r['mean_action']:>8.3f} {r['on_fraction']:>7.2f} "
              f"{r['mean_power']:>9.2f} {r['total_reward']:>11.2f}")

    print(SEP)

    # Mean-only comparison block
    mean_rows = [r for r in rows if r['Episode'] == 'MEAN']
    if len(mean_rows) > 1:
        print('\n\nMEAN VALUES ACROSS ALL EPISODES')
        print(SEP)
        print(f"{'Model':<18} {'ener_tot':>10} {'tdis_tot':>10} {'cost_tot':>10} "
              f"{'emis_tot':>10} {'In-band%':>9} {'OnFrac':>7} "
              f"{'MeanPwr':>9} {'MeanZone°C':>11} {'CumReward':>11}")
        print(SEP)
        for r in mean_rows:
            print(f"  {r['Model']:<16} {r['ener_tot']:>10.4f} {r['tdis_tot']:>10.4f} "
                  f"{r['cost_tot']:>10.4f} {r['emis_tot']:>10.4f} "
                  f"{r['pct_comfortable']:>9.1f} {r['on_fraction']:>7.2f} "
                  f"{r['mean_power']:>9.2f} {r['mean_zone_C']:>11.2f} "
                  f"{r['total_reward']:>11.2f}")

        # Relative differences vs first model
        ref = mean_rows[0]
        print(f'\n\nRELATIVE TO {ref["Model"]} (negative energy/discomfort/cost = improvement)')
        print(SEP)
        print(f"{'Model':<18} {'Δener_tot':>10} {'Δtdis_tot':>10} {'Δcost_tot':>10} "
              f"{'Δemis_tot':>10} {'ΔIn-band%':>10} {'ΔCumReward':>11}")
        print(SEP)
        for r in mean_rows[1:]:
            print(f"  {r['Model']:<16} "
                  f"{r['ener_tot']-ref['ener_tot']:>+10.4f} "
                  f"{r['tdis_tot']-ref['tdis_tot']:>+10.4f} "
                  f"{r['cost_tot']-ref['cost_tot']:>+10.4f} "
                  f"{r['emis_tot']-ref['emis_tot']:>+10.4f} "
                  f"{r['pct_comfortable']-ref['pct_comfortable']:>+10.1f}% "
                  f"{r['total_reward']-ref['total_reward']:>+11.2f}")
        print(SEP)

    print()


def save_summary_files(rows: list, save_dir: str):
    import io

    txt_path = os.path.join(save_dir, 'summary.txt')
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    print_summary_table(rows)
    sys.stdout = old_stdout
    with open(txt_path, 'w') as fh:
        fh.write(buf.getvalue())
    print(f'  Saved: {txt_path}')

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

def run_rl_comparison(model_dirs: list, model_names: list = None,
                      n_episodes: int = 2,
                      url: str = 'http://127.0.0.1:8000',
                      save_dir: str = None):
    """
    Run the full RL-vs-RL comparison pipeline.

    Parameters
    ----------
    model_dirs   : list of 2–3 paths, each containing final_model.zip + config.json
    model_names  : display labels (default: directory basenames)
    n_episodes   : episodes per model (same start times for all)
    url          : BOPTEST REST API base URL
    save_dir     : output directory (default: first model dir / rl_comparison/)
    """
    _require_sb3()

    if not (2 <= len(model_dirs) <= 3):
        raise ValueError(f"Provide 2 or 3 model directories (got {len(model_dirs)})")

    if model_names is None:
        model_names = [os.path.basename(os.path.normpath(d)) for d in model_dirs]
    if len(model_names) != len(model_dirs):
        raise ValueError("--names must have the same number of entries as model directories")

    if save_dir is None:
        save_dir = os.path.join(model_dirs[0], 'rl_comparison')
    os.makedirs(save_dir, exist_ok=True)

    print('=' * 80)
    print('RL MODEL COMPARISON')
    print('=' * 80)
    for name, d in zip(model_names, model_dirs):
        print(f'  {name:<20}  →  {d}')
    print(f'\nOutput directory: {save_dir}')
    print(f'Episodes per model: {n_episodes}')
    print(f'BOPTEST URL: {url}')

    # ── Episode start times ────────────────────────────────────────────────
    # First week of January, then weekly increments.
    start_times = [i * 7 * 86400 for i in range(n_episodes)]
    print(f'\nEpisode start times:')
    for i, s in enumerate(start_times):
        print(f'  Episode {i+1}: day {s // 86400 + 1} of year  ({s:,} s)')

    # ── Load each model and verify its env ────────────────────────────────
    models     = []   # SB3 model instances
    envs       = []   # wrapped envs
    base_envs  = []   # unwrapped BoptestGymEnv
    step_periods = []
    comfort_schedules = []

    for name, model_dir in zip(model_names, model_dirs):
        print(f'\n{"─"*60}')
        print(f'  Loading: {name}')

        config_path = os.path.join(model_dir, 'config.json')
        model_path  = os.path.join(model_dir, 'final_model.zip')

        if not os.path.exists(config_path):
            raise FileNotFoundError(f'config.json not found in {model_dir}')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'final_model.zip not found in {model_dir}')

        with open(config_path) as f:
            config = json.load(f)

        algorithm  = config.get('algorithm', 'SAC')
        testcase   = config.get('testcase', 'bestest_hydronic_heat_pump')
        env_params = config.get('env_params', {})

        # Episode length: 3 days at step_period
        step_period = env_params.get('step_period', 1800)
        vis_steps   = 144   # 3 days × 48 steps/day
        vis_env_params = dict(env_params)
        vis_env_params['max_episode_length'] = vis_steps * step_period

        # Comfort schedule
        raw_sched = env_params.get('comfort_schedule', None)
        if raw_sched:
            comfort_schedule = [(e[0], e[1], tuple(e[2])) for e in raw_sched]
        else:
            comfort_schedule = _DEFAULT_SCHEDULE

        algo_cls = {'SAC': SAC, 'PPO': PPO, 'DQN': DQN}.get(algorithm, SAC)
        rl_model = algo_cls.load(model_path)
        print(f'  Algorithm : {algorithm}')

        env = create_env(
            url=url, testcase=testcase,
            use_custom_reward=True,
            env_params=vis_env_params,
            _register_global=False,
        )

        expected_dim = config.get('observation_dim', env.observation_space.shape[0])
        actual_dim   = env.observation_space.shape[0]
        if actual_dim != expected_dim:
            env.close()
            raise RuntimeError(
                f'Observation space mismatch for "{name}": '
                f'model expects {expected_dim}, env produces {actual_dim}.\n'
                f'Check that the model was trained with the current boptest_gym_env.py.'
            )
        print(f'  Obs space : {env.observation_space.shape}  ✓')

        base_env = env
        while hasattr(base_env, 'env'):
            base_env = base_env.env

        models.append(rl_model)
        envs.append(env)
        base_envs.append(base_env)
        step_periods.append(step_period)
        comfort_schedules.append(comfort_schedule)

    # ── Run episodes ───────────────────────────────────────────────────────
    all_data = {n: [] for n in model_names}

    for idx, (name, rl_model, env, base_env, step_period, comfort_schedule) in enumerate(
            zip(model_names, models, envs, base_envs, step_periods, comfort_schedules)):

        print(f'\n{"═"*80}')
        print(f'RUNNING: {name}')
        print(f'{"═"*80}')

        for ep in range(n_episodes):
            base_env.start_time = start_times[ep]
            base_env.random_start_time = False

            d = run_rl_episode(
                env=env,
                model=rl_model,
                base_env=base_env,
                episode_num=ep + 1,
                comfort_schedule=comfort_schedule,
                step_period=step_period,
                model_label=name,
            )
            all_data[name].append(d)

        env.close()
        print(f'\n✓ {name} episodes complete')

    # ── Plots ──────────────────────────────────────────────────────────────
    print(f'\n{"═"*80}')
    print('GENERATING COMPARISON PLOTS')
    print(f'{"═"*80}')

    for ep in range(n_episodes):
        plot_combined_episode(all_data, model_names, save_dir, episode_num=ep + 1)

    if n_episodes > 1:
        plot_multi_episode_summary(all_data, model_names, save_dir)

    # ── Summary ────────────────────────────────────────────────────────────
    print(f'\n{"═"*80}')
    rows = compute_summary(all_data, model_names)
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
        description='Compare 2–3 trained RL models on the same BOPTEST episodes.'
    )
    parser.add_argument('model_dirs', nargs='+', metavar='MODEL_DIR',
                        help='2 or 3 directories, each with final_model.zip + config.json')
    parser.add_argument('--names', type=str, default=None,
                        help='Comma-separated display names, e.g. "SAC-v1,SAC-v2,SAC-v3"')
    parser.add_argument('--episodes', type=int, default=2,
                        help='Episodes per model (default: 2)')
    parser.add_argument('--url', type=str, default='http://127.0.0.1:8000',
                        help='BOPTEST server URL (default: http://127.0.0.1:8000)')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Output directory (default: <first_model_dir>/rl_comparison/)')

    args = parser.parse_args()

    if not (2 <= len(args.model_dirs) <= 3):
        print(f'Error: provide 2 or 3 model directories (got {len(args.model_dirs)})')
        sys.exit(1)

    for d in args.model_dirs:
        if not os.path.isdir(d):
            print(f'Error: directory not found: {d}')
            sys.exit(1)

    model_names = None
    if args.names:
        model_names = [n.strip() for n in args.names.split(',')]
        if len(model_names) != len(args.model_dirs):
            print(f'Error: --names has {len(model_names)} entries but '
                  f'{len(args.model_dirs)} model directories were given')
            sys.exit(1)

    run_rl_comparison(
        model_dirs=args.model_dirs,
        model_names=model_names,
        n_episodes=args.episodes,
        url=args.url,
        save_dir=args.save_dir,
    )
