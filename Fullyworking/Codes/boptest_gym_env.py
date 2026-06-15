"""
BOPTEST Gymnasium Environment
Recreated and improved implementation based on original boptestGymEnv.py
"""

import time
import gymnasium as gym
import requests
import numpy as np
import random
from requests.exceptions import RequestException
from gymnasium import spaces


# ── Module-level utility ──────────────────────────────────────────────────────

def extract_value(data, key, default=0):
    """Extract a numeric value from a BOPTEST measurement dict.

    BOPTEST can return measurements either as plain scalars or as
    ``{"value": X, "unit": Y}`` dicts.  This function handles both.

    Parameters
    ----------
    data : dict
        Measurement payload from BOPTEST.
    key : str
        Key to look up.
    default : float or None
        Value to return when the key is absent or the nested 'value' key is
        absent.  ``None`` is treated as ``0`` to prevent ``float(None)``
        raising a ``TypeError``.
    """
    # Normalise None default so float() never receives None
    _default = 0 if default is None else default
    val = data.get(key, _default)
    if isinstance(val, dict):
        v = val.get('value', _default)
        return float(_default if v is None else v)
    if val is None:
        return float(_default)
    return float(val)


class BoptestGymEnv(gym.Env):
    """
    BOPTEST Environment that follows Gymnasium interface.
    Allows RL agents to interact with building emulator models from BOPTEST.
    """

    metadata = {'render_modes': ['console']}
    DEFAULT_RUNTIME_TIMEOUT = 30
    DEFAULT_RUNTIME_RETRIES = 2

    def __init__(self,
                 url='http://127.0.0.1:8000',
                 testcase='bestest_hydronic_heat_pump',
                 actions=['oveHeaPumY_u'],
                 observations={'reaTZon_y': (280., 310.)},
                 max_episode_length=24*3600,
                 random_start_time=False,
                 excluding_periods=None,
                 regressive_period=None,
                 predictive_period=None,
                 predictive_step=None,
                 start_time=0,
                 warmup_period=0,
                 scenario={'electricity_price': 'constant'},
                 step_period=3600):
        """
        Parameters
        ----------
        url : str
            REST API url for BOPTEST interface
        testcase : str
            Testcase identifier
        actions : list
            List of action variable names
        observations : dict
            Dictionary mapping observation keys to (min, max) bounds
        max_episode_length : int
            Maximum episode duration in seconds
        random_start_time : bool
            Use random start time for each episode
        excluding_periods : list of tuples
            Periods to exclude from random start times
        regressive_period : int
            Seconds for historical observations
        predictive_period : int
            Seconds for forecast observations
        predictive_step : int, optional
            Interval between forecast steps in seconds.  Defaults to
            ``step_period`` (same resolution as simulation steps).  Set to
            a multiple of ``step_period`` to look further ahead with the
            same number of observation slots (e.g. 2 × step_period gives
            t+0, t+60 min, t+120 min, t+180 min with 4 slots).
        start_time : int
            Fixed episode start time (seconds from year start)
        warmup_period : int
            Warmup period before episode starts
        scenario : dict
            BOPTEST scenario configuration
        step_period : int
            Time step in seconds
        """

        super().__init__()

        self.url = url
        self.testcase = testcase
        self.actions_list = actions
        self.observations_dict = observations
        self.max_episode_length = max_episode_length
        self.random_start_time = random_start_time
        self.excluding_periods = excluding_periods
        self.start_time = start_time
        self.warmup_period = warmup_period
        self.predictive_period = predictive_period
        # Normalise 0 → None: "0 steps of history" is the same as "no history",
        # but 0 (int) would pass the `is not None` guard and crash on pop() from
        # empty observation_history lists.  JSON configs save 0 for no-regression.
        self.regressive_period = regressive_period if regressive_period else None
        self.step_period = step_period
        self.predictive_step = predictive_step if predictive_step is not None else step_period
        self.scenario = scenario

        # Margins to avoid year boundaries
        if self.regressive_period is not None:
            self.bgn_year_margin = self.regressive_period
        else:
            self.bgn_year_margin = 0
        self.end_year_margin = self.max_episode_length

        # Initialize BOPTEST connection
        self._initialize_boptest()

        # Define action and observation spaces
        self._define_spaces()

        # Initialize episode tracking
        self.episode_reward = 0
        self.objective_integrand = 0
        self.res = None
        self.last_forecast = {}     # populated after first _get_observation()
        self.last_measurements = {} # cached from _get_observation(); read by CustomRewardWrapper
        self.last_kpis = {}         # cached from get_reward()/reset(); read by CustomRewardWrapper

    def _api_call(self, method, url, json=None, timeout=None, retries=0, context='API call'):
        """Call a BOPTEST endpoint with retries and status validation."""
        method = method.lower()
        timeout = self.DEFAULT_RUNTIME_TIMEOUT if timeout is None else timeout
        if method == 'get':
            caller = requests.get
        elif method == 'post':
            caller = requests.post
        elif method == 'put':
            caller = requests.put
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

        last_error = None
        for attempt in range(retries + 1):
            try:
                if json is None:
                    response = caller(url, timeout=timeout)
                else:
                    response = caller(url, json=json, timeout=timeout)
                if response.status_code != 200:
                    raise RuntimeError(
                        f"{context} failed (status {response.status_code}): {url}"
                    )
                return response
            except (RequestException, RuntimeError, ValueError) as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(0.5)
                else:
                    raise RuntimeError(f"{context} failed after {retries + 1} attempt(s): {exc}")
        raise RuntimeError(f"{context} failed: {last_error}")

    @staticmethod
    def _get_payload(response, context='API response'):
        """Extract payload while validating JSON structure."""
        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(f"{context} returned invalid JSON: {exc}")
        if isinstance(data, dict):
            return data.get('payload', data)
        raise RuntimeError(f"{context} returned unexpected JSON type: {type(data)}")

    def _initialize_boptest(self):
        """Initialize connection to BOPTEST and get test information."""

        # Stop this instance's previous test if it has one (multi-instance safe:
        # only stops the testid owned by this object, not all running instances).
        try:
            if hasattr(self, 'testid') and self.testid:
                requests.put(f'{self.url}/stop/{self.testid}', timeout=2)
        except Exception:
            pass

        # Select testcase with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f'{self.url}/testcases/{self.testcase}/select',
                    timeout=30
                )
                if response.status_code != 200:
                    raise RuntimeError(f"Failed to select testcase: {response.status_code}")

                self.testid = response.json()['testid']
                print(f"Selected testcase: {self.testcase}, testid: {self.testid}")
                break

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    print(f"Timeout on attempt {attempt + 1}/{max_retries}, retrying...")
                    time.sleep(5)
                else:
                    raise RuntimeError(
                        "BOPTEST server timeout - it may be stuck.\n"
                        "Try running: python cleanup_boptest.py"
                    )
            except requests.exceptions.ConnectionError:
                if attempt < max_retries - 1:
                    print(f"Connection failed on attempt {attempt + 1}/{max_retries}, retrying...")
                    time.sleep(5)
                else:
                    raise RuntimeError(
                        f"Cannot connect to BOPTEST server at {self.url}.\n"
                        "Is Docker running? Try: docker ps | grep boptest"
                    )

        # Get testcase information
        try:
            meas_resp = requests.get(f'{self.url}/measurements/{self.testid}', timeout=10)
            fc_resp = requests.get(f'{self.url}/forecast_points/{self.testid}', timeout=10)
            inp_resp = requests.get(f'{self.url}/inputs/{self.testid}', timeout=10)
            if meas_resp.status_code != 200 or fc_resp.status_code != 200 or inp_resp.status_code != 200:
                raise RuntimeError(
                    "Failed to fetch testcase metadata: "
                    f"measurements={meas_resp.status_code}, "
                    f"forecast_points={fc_resp.status_code}, "
                    f"inputs={inp_resp.status_code}"
                )
            self.all_measurement_vars = meas_resp.json()['payload']
            self.all_predictive_vars = fc_resp.json()['payload']
            self.all_input_vars = inp_resp.json()['payload']
        except requests.exceptions.Timeout:
            raise RuntimeError(
                "BOPTEST server timeout during initialization.\n"
                "Try running: python cleanup_boptest.py"
            )

        # Set step period
        step_resp = requests.put(
            f'{self.url}/step/{self.testid}',
            json={'step': self.step_period},
            timeout=30,
        )
        if step_resp.status_code != 200:
            raise RuntimeError(f"Failed to set step period: {step_resp.status_code}")

        # Set scenario (can take longer for complex testcases like multizone)
        if self.scenario:
            print(f"Setting scenario: {self.scenario}")
            print("This may take 30-60 seconds for complex testcases...")
            try:
                scenario_resp = requests.put(
                    f'{self.url}/scenario/{self.testid}',
                    json=self.scenario,
                    timeout=120
                )
                if scenario_resp.status_code == 200:
                    print("✓ Scenario set successfully")
                else:
                    print(f"⚠ Scenario response: {scenario_resp.status_code}")
            except requests.exceptions.Timeout:
                print("⚠ Warning: Scenario setting timed out (may still be processing)")
                print("   Continuing anyway - simulation should still work")

        self._detect_power_vars()

    def _detect_power_vars(self):
        """Auto-detect power measurement variable names for the current testcase.

        Different BOPTEST testcases use different variable names for electrical
        power (e.g. ``reaPHeaPum_y`` vs ``PHea``).  This method builds the
        ``power_measurement_vars`` dict that visualisation code relies on.
        """
        candidates = {
            'heat_pump': ['reaPHeaPum_y', 'reaPHeatPum_y', 'PHea', 'PHeaPum'],
            'fan':       ['reaPFan_y', 'reaPFan', 'PFan'],
            'pump':      ['reaPPumEmi_y', 'reaPPump_y', 'PPumEmi', 'PPum'],
            'total':     ['reaPTot_y', 'PTot'],
        }
        self.power_measurement_vars = {}
        for component, names in candidates.items():
            for var in names:
                if var in self.all_measurement_vars:
                    self.power_measurement_vars[component] = var
                    break

    def _define_spaces(self):
        """Define Gymnasium action and observation spaces."""

        # Action space
        action_lows = []
        action_highs = []

        for action_name in self.actions_list:
            if action_name in self.all_input_vars:
                action_lows.append(self.all_input_vars[action_name]['Minimum'])
                action_highs.append(self.all_input_vars[action_name]['Maximum'])
            else:
                action_lows.append(0.0)
                action_highs.append(1.0)
                print(f"Warning: Action '{action_name}' not found, using default bounds [0, 1]")

        self.action_space = spaces.Box(
            low=np.array(action_lows, dtype=np.float32),
            high=np.array(action_highs, dtype=np.float32),
            dtype=np.float32
        )

        # Observation space
        obs_lows = []
        obs_highs = []
        self.obs_keys = []

        for obs_name, (low, high) in self.observations_dict.items():
            is_measurement = obs_name in self.all_measurement_vars
            is_predictive = obs_name in self.all_predictive_vars

            if not is_measurement and not is_predictive and obs_name != 'time':
                print(f"Warning: '{obs_name}' not found in measurements or forecasts")

            # Add predictive observations
            if self.predictive_period is not None and (is_predictive or obs_name == 'time'):
                n_steps = int(self.predictive_period / self.predictive_step)
                if self.predictive_period == 0:
                    n_steps = 1

                for i in range(n_steps):
                    obs_lows.append(low)
                    obs_highs.append(high)
                    self.obs_keys.append(f"{obs_name}_pred_{i}")

            # Add current measurement
            if is_measurement or obs_name == 'time':
                obs_lows.append(low)
                obs_highs.append(high)
                self.obs_keys.append(obs_name)

            # Add regressive observations
            if self.regressive_period is not None and is_measurement:
                n_steps = int(self.regressive_period / self.step_period)
                for i in range(1, n_steps + 1):
                    obs_lows.append(low)
                    obs_highs.append(high)
                    self.obs_keys.append(f"{obs_name}_reg_{i}")

        self.observation_space = spaces.Box(
            low=np.array(obs_lows, dtype=np.float32),
            high=np.array(obs_highs, dtype=np.float32),
            dtype=np.float32
        )

        # Initialize history for regressive period
        if self.regressive_period is not None:
            n_steps = int(self.regressive_period / self.step_period)
            self.observation_history = {
                obs_name: [None] * n_steps
                for obs_name in self.observations_dict.keys()
                if obs_name in self.all_measurement_vars
            }

    def reset(self, seed=None, options=None):
        """Reset the environment to start a new episode."""

        super().reset(seed=seed)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Determine start time
        if self.random_start_time:
            self.start_time = self._sample_start_time()

        # Initialize episode
        init_payload = {
            'start_time': int(self.start_time),
            'warmup_period': int(self.warmup_period)
        }
        response = self._api_call(
            'put',
            f'{self.url}/initialize/{self.testid}',
            json=init_payload,
            timeout=self.DEFAULT_RUNTIME_TIMEOUT,
            retries=self.DEFAULT_RUNTIME_RETRIES,
            context='Initialize episode',
        )
        self.res = self._get_payload(response, 'Initialize episode')

        # Reset tracking variables
        self.episode_reward = 0

        # Get initial KPIs for reward calculation
        kpi_resp = self._api_call(
            'get',
            f'{self.url}/kpi/{self.testid}',
            timeout=self.DEFAULT_RUNTIME_TIMEOUT,
            retries=self.DEFAULT_RUNTIME_RETRIES,
            context='Fetch KPIs on reset',
        )
        kpis = self._get_payload(kpi_resp, 'Fetch KPIs on reset')
        self.last_kpis = kpis  # cache so CustomRewardWrapper.reset() needs no second GET /kpi/
        self.objective_integrand = kpis['cost_tot'] + kpis['tdis_tot']

        # Initialize observation history
        if self.regressive_period is not None:
            for key in self.observation_history:
                self.observation_history[key] = [None] * len(self.observation_history[key])

        # Get initial observation
        obs = self._get_observation()

        return obs, {}

    def _sample_start_time(self):
        """Sample random start time avoiding excluded periods.

        Raises ``RuntimeError`` after ``max_attempts`` tries so the caller
        never blocks in an infinite loop when ``excluding_periods`` covers the
        entire valid year range.
        """

        year_seconds = 365 * 24 * 3600
        max_attempts = 10_000

        for _ in range(max_attempts):
            start = random.randint(
                self.bgn_year_margin,
                year_seconds - self.end_year_margin
            )

            if self.excluding_periods is None:
                return start

            valid = True
            for excl_start, excl_end in self.excluding_periods:
                episode_end = start + self.max_episode_length
                if not (episode_end < excl_start or start > excl_end):
                    valid = False
                    break

            if valid:
                return start

        raise RuntimeError(
            f"Could not find a valid random start time after {max_attempts} "
            "attempts. The excluding_periods may cover the entire valid year range."
        )

    def _get_observation(self):
        """Get current observation vector."""

        obs_list = []

        # POST /advance/ and PUT /initialize/ both return the full measurement
        # payload in self.res — use it directly to avoid a redundant
        # GET /measurements/ call (saves 1 HTTP round-trip every step,
        # reducing total HTTP calls by 25% during training).
        measurements = dict(self.res) if self.res else {}
        self.last_measurements = measurements

        current_time = extract_value(measurements, 'time', 0)

        # Get forecast if needed
        forecast_data = {}
        if self.predictive_period is not None:
            forecast_vars = [
                name for name in self.observations_dict.keys()
                if name in self.all_predictive_vars
            ]

            if forecast_vars:
                forecast_payload = {
                    'point_names': forecast_vars,
                    'horizon': int(self.predictive_period),
                    'interval': int(self.predictive_step)
                }
                response = self._api_call(
                    'put',
                    f'{self.url}/forecast/{self.testid}',
                    json=forecast_payload,
                    timeout=self.DEFAULT_RUNTIME_TIMEOUT,
                    retries=self.DEFAULT_RUNTIME_RETRIES,
                    context='Fetch forecast',
                )
                forecast_data = self._get_payload(response, 'Fetch forecast')
                if isinstance(forecast_data, dict):
                    self.last_forecast = forecast_data

        # Build observation vector
        for obs_name, (low, high) in self.observations_dict.items():
            is_measurement = obs_name in self.all_measurement_vars
            is_predictive = obs_name in self.all_predictive_vars

            # Predictive observations
            if self.predictive_period is not None and (is_predictive or obs_name == 'time'):
                n_steps = int(self.predictive_period / self.predictive_step)
                if self.predictive_period == 0:
                    n_steps = 1

                if obs_name == 'time':
                    for i in range(n_steps):
                        time_val = (current_time + i * self.predictive_step) % (high - low) + low
                        obs_list.append(float(time_val))
                elif obs_name in forecast_data:
                    forecast_vals = forecast_data[obs_name]
                    for i in range(n_steps):
                        if i < len(forecast_vals):
                            obs_list.append(float(forecast_vals[i]))
                        else:
                            current_val = extract_value(measurements, obs_name, (low + high) / 2)
                            obs_list.append(current_val)
                else:
                    current_val = extract_value(measurements, obs_name, (low + high) / 2)
                    obs_list.extend([current_val] * n_steps)

            # Current measurement
            if is_measurement or obs_name == 'time':
                if obs_name == 'time':
                    time_val = current_time % (high - low) + low
                    obs_list.append(float(time_val))
                else:
                    current_val = extract_value(measurements, obs_name, (low + high) / 2)
                    obs_list.append(current_val)

                    # Update history
                    if self.regressive_period is not None and obs_name in self.observation_history:
                        self.observation_history[obs_name].pop()
                        self.observation_history[obs_name].insert(0, current_val)

            # Regressive observations
            if self.regressive_period is not None and is_measurement:
                for hist_val in self.observation_history[obs_name]:
                    if hist_val is not None:
                        obs_list.append(float(hist_val))
                    else:
                        current_val = extract_value(measurements, obs_name, (low + high) / 2)
                        obs_list.append(current_val)

        return np.array(obs_list, dtype=np.float32)

    def step(self, action):
        """Execute one step in the environment."""

        # Convert action to dictionary and add activation signals
        action_dict = {}
        for i, action_name in enumerate(self.actions_list):
            action_dict[action_name] = float(action[i])

            # BOPTEST requires activation signal for overrides
            if action_name.startswith('ove'):
                activate_name = action_name.replace('_u', '_activate')
                action_dict[activate_name] = 1

        # Advance simulation
        response = self._api_call(
            'post',
            f'{self.url}/advance/{self.testid}',
            json=action_dict,
            timeout=self.DEFAULT_RUNTIME_TIMEOUT,
            retries=self.DEFAULT_RUNTIME_RETRIES,
            context='Advance simulation',
        )
        self.res = self._get_payload(response, 'Advance simulation')

        # Get observation
        obs = self._get_observation()

        # Calculate reward
        reward = self.get_reward()
        self.episode_reward += reward

        # Check if episode is done
        terminated = False
        truncated = self.compute_truncated()

        sim_time = extract_value(self.res, 'time', 0)

        info = {
            'time': sim_time,
            'episode_reward': self.episode_reward
        }

        return obs, reward, terminated, truncated, info

    def get_reward(self):
        """Calculate reward for current step."""

        response = self._api_call(
            'get',
            f'{self.url}/kpi/{self.testid}',
            timeout=self.DEFAULT_RUNTIME_TIMEOUT,
            retries=self.DEFAULT_RUNTIME_RETRIES,
            context='Fetch KPIs for reward',
        )
        kpis = self._get_payload(response, 'Fetch KPIs for reward')
        self.last_kpis = kpis  # cache so CustomRewardWrapper needs no second GET /kpi/

        objective_integrand = kpis['cost_tot'] + kpis['tdis_tot']

        reward = -(objective_integrand - self.objective_integrand)

        self.objective_integrand = objective_integrand

        return reward

    def compute_truncated(self):
        """Check if episode should be truncated."""
        sim_time = extract_value(self.res, 'time', 0) if self.res else 0
        elapsed_time = sim_time - (self.start_time + self.warmup_period)
        return elapsed_time >= self.max_episode_length

    def get_kpis(self):
        """Get KPIs from BOPTEST."""

        response = self._api_call(
            'get',
            f'{self.url}/kpi/{self.testid}',
            timeout=self.DEFAULT_RUNTIME_TIMEOUT,
            retries=self.DEFAULT_RUNTIME_RETRIES,
            context='Fetch KPIs',
        )
        payload = self._get_payload(response, 'Fetch KPIs')
        return payload if isinstance(payload, dict) else {}

    def stop(self):
        """Stop the current test."""

        if hasattr(self, 'testid') and self.testid:
            try:
                response = requests.put(
                    f'{self.url}/stop/{self.testid}',
                    timeout=5,
                )
                if response.status_code == 200:
                    print(f"✅ Stopped test {self.testid} - Worker is free")
                else:
                    print(f"⚠ Stop returned status {response.status_code}")
            except requests.exceptions.Timeout:
                print("⚠ Stop request timed out - BOPTEST may be stuck")
            except Exception as e:
                print(f"⚠ Error stopping test: {e}")

    def close(self):
        """Close the environment and cleanup.

        Only stops this instance's testid — safe to call when other
        BOPTEST instances are running on the same server.
        """
        self.stop()

    def __del__(self):
        """Destructor to ensure cleanup on deletion."""
        try:
            self.close()
        except Exception:
            pass


class NormalizedObservationWrapper(gym.ObservationWrapper):
    """Normalize observations to [0, 1]."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=env.observation_space.shape,
            dtype=np.float32
        )
        self.obs_low = env.observation_space.low
        self.obs_high = env.observation_space.high

    def observation(self, obs):
        """Normalize observation."""
        normalized = (obs - self.obs_low) / (self.obs_high - self.obs_low + 1e-8)
        return np.clip(normalized, 0, 1).astype(np.float32)


class DiscretizedActionWrapper(gym.ActionWrapper):
    """Discretize continuous action space."""

    def __init__(self, env, n_bins_act=10):
        super().__init__(env)
        self.n_bins_act = n_bins_act
        self.n_actions = env.action_space.shape[0]
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high

        # Single discrete action space
        self.action_space = spaces.Discrete(n_bins_act + 1)

    def action(self, action):
        """Convert discrete action to continuous."""
        normalized = action / self.n_bins_act
        continuous = self.action_low + normalized * (self.action_high - self.action_low)
        return continuous


class RewardWeightDiscomfort(BoptestGymEnv):
    """Environment with modified reward to weight discomfort more heavily."""

    def __init__(self, discomfort_weight=10, **kwargs):
        super().__init__(**kwargs)
        self.discomfort_weight = discomfort_weight

    def get_reward(self):
        """Custom reward with weighted discomfort."""

        kpis = requests.get(f'{self.url}/kpi/{self.testid}').json()['payload']

        objective_integrand = kpis['cost_tot'] + self.discomfort_weight * kpis['tdis_tot']

        reward = -(objective_integrand - self.objective_integrand)

        self.objective_integrand = objective_integrand

        return reward


class CustomRewardWrapper(gym.Wrapper):
    """
    Custom reward wrapper that balances thermal comfort and energy savings.

    Supports a dynamic day/night comfort schedule so the reward correctly
    penalises deviations from time-varying setpoints.  When
    ``comfort_schedule`` is ``None`` the fixed ``comfort_temp_range`` is used.

    Reward function (Huber-Band)
    ----------------------------
    comfort_reward = 0.0                                        when inside band (neutral)
                   = -comfort_weight * (e_c² / (2·δ))          when 0 < e_c ≤ huber_delta  (quadratic)
                   = -comfort_weight * (e_c − δ/2)             when e_c > huber_delta      (linear)
                     e_c = max(0, T_lower − T_zone, T_zone − T_upper)  (K, always ≥ 0)
                     δ   = huber_delta  (K, default 0.25)
    price_reward   = price_weight * (-(cost_step / max_price_per_step))
                     cost_step        = energy_kwh * constant_price  (EUR/m²/step)
                     energy_kwh       = ener_tot delta from KPI  (kWh/m²/step)
                     constant_price   = fixed electricity price  (EUR/kWh)
                     max_price_per_step = normalisation constant (EUR/m²/step)
    reward         = comfort_reward + price_reward

    The Huber-band loss is smooth and differentiable everywhere: quadratic
    near the band edge (gentle gradient for small violations) and linear
    far outside (bounded gradient prevents exploding penalties).  The neutral
    zero inside the band removes the spurious +1 bias of the previous formula.

    Using energy × constant_price instead of cost_tot ensures the price
    penalty is truly constant and independent of the server's scenario setting.

    ``cliff_reward`` and ``max_pen`` are retained for backward compatibility
    but have no effect on the reward computation.
    """

    # Default day/night comfort schedule.
    # Format: list of (start_hour, end_hour, (lower_K, upper_K))
    # Ranges that wrap around midnight use end_hour < start_hour.
    # Night band is 2 °C wide; day band is 3 °C wide (wider tolerance for occupants).
    # At day-start the zone has drifted to night levels (20–22 °C) and must reach 21–24 °C.
    DEFAULT_COMFORT_SCHEDULE = [
        # Occupied day hours  07:00–22:00 → 21–24 °C  (3 °C band)
        (7, 22, (294.15, 297.15)),
        # Unoccupied night hours  22:00–07:00 → 20–22 °C  (2 °C band, setback)
        (22, 7,  (293.15, 295.15)),
    ]

    def __init__(
        self,
        env,
        comfort_weight=50.0,
        price_weight=30.0,
        max_pen=20.0,
        comfort_temp_range=(294.15, 297.15),  # 21-24°C default (day, 3°C band)
        comfort_schedule=None,
        cliff_reward=True,                    # retained for backward compat; unused
        max_price_per_step=0.01,              # normalisation: max expected EUR/m²/step
        constant_price=0.1,                   # fixed electricity price (EUR/kWh)
        huber_delta=0.25,                     # Huber transition (K): ≤δ quadratic, >δ linear
    ):
        super().__init__(env)
        self.comfort_weight = comfort_weight
        self.cliff_reward = cliff_reward
        self.price_weight = price_weight
        self.max_pen = max_pen
        self.max_price_per_step = max_price_per_step
        self.constant_price = constant_price
        self.comfort_temp_range = comfort_temp_range
        self.huber_delta = huber_delta
        self.comfort_schedule = (
            comfort_schedule if comfort_schedule is not None
            else self.DEFAULT_COMFORT_SCHEDULE
        )

        self.prev_kpi_energy = None
        self.timestep_seconds = getattr(env, 'step_period', 3600)

    def _has_valid_zone_temp(self, measurements):
        """Return True when *measurements* contains a readable zone temperature.

        A value is considered readable if it is:
        - a plain numeric scalar, **or**
        - a ``{'value': X, 'unit': Y}`` dict with a non-None ``'value'`` key.

        A metadata dict (``{'Minimum': X, 'Maximum': Y, 'unit': Z}``) that
        has no ``'value'`` key is considered *not* readable and returns False.
        This is used to decide whether to fall back to ``base_env.res``.
        """
        for var in ('reaTZon_y', 'reaTRooAir_y', 'TRooAir_y', 'TZon_y'):
            val = measurements.get(var)
            if val is None:
                continue
            if isinstance(val, dict):
                if 'value' in val and val['value'] is not None:
                    return True
            else:
                return True  # plain scalar (int / float)
        return False

    def _get_zone_temp(self, measurements):
        """Return the zone temperature (K) from a measurements payload.

        Tries common variable names in priority order so the wrapper works
        across multiple BOPTEST testcases without extra configuration.
        Falls back to 293.15 K (~20 °C) when nothing is found.

        This avoids passing ``None`` as a default to :func:`extract_value`,
        which would cause a ``float(None)`` ``TypeError``.
        """
        for var in ('reaTZon_y', 'reaTRooAir_y', 'TRooAir_y', 'TZon_y'):
            val = measurements.get(var)
            if val is None:
                continue
            if isinstance(val, dict):
                v = val.get('value')
                if v is not None:
                    return float(v)
            else:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    pass
        return 293.15  # fallback: ~20 °C


    def _get_comfort_range(self, sim_time):
        """Return ``(lower_K, upper_K)`` comfort bounds for the given simulation
        time (seconds from year start).

        When ``comfort_schedule`` is ``None``, the fixed
        ``comfort_temp_range`` is returned.  Otherwise the schedule is
        searched for the matching time window; if no window matches the fixed
        ``comfort_temp_range`` is used as a fallback.
        """
        if self.comfort_schedule is None:
            return self.comfort_temp_range

        seconds_per_day = 86400.0
        hour_of_day = (sim_time % seconds_per_day) / 3600.0

        for start_h, end_h, bounds in self.comfort_schedule:
            lower, upper = float(bounds[0]), float(bounds[1])
            if start_h < end_h:
                # Normal range: e.g. 7 → 22
                if start_h <= hour_of_day < end_h:
                    return (lower, upper)
            else:
                # Wrap-around range: e.g. 22 → 7 (crosses midnight)
                if hour_of_day >= start_h or hour_of_day < end_h:
                    return (lower, upper)

        # Fallback to fixed range
        return self.comfort_temp_range

    def reset(self, **kwargs):
        """Reset and initialize KPI tracking."""
        obs, info = self.env.reset(**kwargs)

        # BoptestGymEnv.reset() already fetched KPIs and cached them in last_kpis.
        # Read from the cache instead of making a second GET /kpi/ call.
        try:
            base = self.env
            while hasattr(base, 'env'):
                base = base.env
            kpis = getattr(base, 'last_kpis', {})
            self.prev_kpi_energy = kpis.get('ener_tot', 0)
        except Exception:
            self.prev_kpi_energy = 0

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Navigate to BoptestGymEnv to get cached measurements and KPIs.
        # self.res on BoptestGymEnv is ALWAYS in value format (scalar or
        # {'value': X, 'unit': Y} dict) because it comes directly from
        # POST /advance/ or PUT /initialize/.  We use it as a fallback
        # when GET /measurements returns metadata-format dicts (no 'value').
        _base_env = self.env
        while hasattr(_base_env, 'env'):
            _base_env = _base_env.env
        _base_res = getattr(_base_env, 'res', None) or {}

        # Read measurements from cache — BoptestGymEnv._get_observation() already
        # fetched and enriched them this step; no second GET /measurements/ needed.
        measurements = getattr(_base_env, 'last_measurements', {})

        # Read KPIs from cache — BoptestGymEnv.get_reward() already fetched them
        # this step; no second GET /kpi/ needed.
        kpis = getattr(_base_env, 'last_kpis', {})

        # ── Zone temperature ──────────────────────────────────────────────
        # Use helper to avoid passing None to extract_value (TypeError).
        # If measurements only contain metadata-format dicts (no 'value'),
        # fall back to the advance response which is always value format.
        zone_temp = self._get_zone_temp(measurements)
        if not self._has_valid_zone_temp(measurements) and _base_res:
            zone_from_res = self._get_zone_temp(_base_res)
            if zone_from_res != 293.15:   # advance response had a real value
                zone_temp = zone_from_res

        # ── Power (W/m²) ──────────────────────────────────────────────────
        # Derived from ener_tot KPI delta — consequence of the action taken.
        current_kpi_energy = kpis.get('ener_tot', 0)
        if self.prev_kpi_energy is not None:
            energy_kwh = current_kpi_energy - self.prev_kpi_energy
            timestep_hours = self.timestep_seconds / 3600
            total_power = (energy_kwh * 1000) / timestep_hours if timestep_hours > 0 else 0
        else:
            energy_kwh = 0.0
            total_power = 0
        self.prev_kpi_energy = current_kpi_energy

        # ── Comfort reward: Huber-band loss ───────────────────────────────
        # e_c = deviation from nearest comfort band edge (0.0 when inside band).
        # Below huber_delta: quadratic penalty (smooth near boundary).
        # Above huber_delta: linear penalty (bounded gradient for large violations).
        # Reward is 0.0 inside the band — no spurious flat +1 bias.
        sim_time = float(info.get('time', 0))
        temp_lower, temp_upper = self._get_comfort_range(sim_time)

        e_c = max(0.0, temp_lower - zone_temp, zone_temp - temp_upper)
        delta = self.huber_delta
        if e_c <= delta:
            comfort_penalty = (e_c ** 2) / (2.0 * delta)    # quadratic near band edge
        else:
            comfort_penalty = e_c - delta / 2.0              # linear far outside
        comfort_reward = -self.comfort_weight * comfort_penalty
        thermal_comfort = (e_c == 0.0)

        # ── Price reward: energy × constant price, normalised ─────────────
        # Uses energy × fixed price instead of cost_tot KPI so the penalty
        # is guaranteed constant regardless of server scenario setting.
        cost_step = energy_kwh * self.constant_price          # EUR/m²/step
        safe_max_price = max(float(self.max_price_per_step), 1e-9)
        price_reward = self.price_weight * (-(cost_step / safe_max_price))

        electricity_price = self.constant_price               # always fixed

        # ── Combined reward ───────────────────────────────────────────────
        reward = comfort_reward + price_reward

        info['comfort_reward']       = comfort_reward
        info['price_reward']         = price_reward
        info['cost_step']            = cost_step
        info['electricity_price']    = electricity_price
        info['thermal_comfort']      = thermal_comfort
        info['zone_temp']            = zone_temp
        info['total_power']          = total_power
        info['kpi_energy']           = current_kpi_energy
        info['comfort_temp_lower_K'] = temp_lower
        info['comfort_temp_upper_K'] = temp_upper

        return obs, reward, terminated, truncated, info


# ── PBRSRewardWrapper ─────────────────────────────────────────────────────────

class PBRSRewardWrapper(gym.Wrapper):
    """Potential-Based Reward Shaping (PBRS) comfort + energy reward.

    Implements Ng, Harada & Russell (1999) policy-invariant reward shaping::

        reward  = r_base + w_c × F
        F       = γ × Φ(s') − Φ(s)           (shaping term)
        Φ(s)    = −(T_zone − T_center)²       (higher = closer to band centre)
        T_center = (T_lower + T_upper) / 2    (midpoint of comfort band at t)
        r_base  = −w_e × (ΔE / E_ref)         (normalised energy penalty)
        ΔE      = ener_tot delta per step (kWh/m²/step, from BOPTEST KPI)

    Properties
    ----------
    - Policy-invariant: F cannot corrupt the optimal policy (Ng 1999 theorem).
    - Dense inside-band gradient toward the band centre (Huber-band has none).
    - Anticipates schedule transitions via the time-varying T_center signal.
    - gamma MUST equal SAC's gamma; mismatch weakens the invariance guarantee.
    """

    DEFAULT_COMFORT_SCHEDULE = [
        (7,  22, (294.15, 297.15)),   # day   07:00–22:00 → 21–24 °C
        (22,  7, (293.15, 295.15)),   # night 22:00–07:00 → 20–22 °C
    ]

    def __init__(
        self,
        env,
        gamma=0.995,                          # discount factor — MUST match SAC gamma
        comfort_weight=10.0,                  # w_c: shaping term scale
        energy_weight=1.0,                    # w_e: energy penalty scale
        E_ref=0.01,                           # kWh/m²/step normalisation constant
        comfort_schedule=None,                # None → DEFAULT_COMFORT_SCHEDULE
        comfort_temp_range=(294.15, 297.15),  # fallback when no schedule entry matches
    ):
        super().__init__(env)
        self.gamma = gamma
        self.comfort_weight = comfort_weight
        self.energy_weight = energy_weight
        self.E_ref = E_ref
        self.comfort_temp_range = comfort_temp_range
        self.comfort_schedule = (
            comfort_schedule if comfort_schedule is not None
            else self.DEFAULT_COMFORT_SCHEDULE
        )
        # State carried across steps
        self.prev_zone_temp  = None
        self.prev_sim_time   = None
        self.prev_kpi_energy = None
        self.timestep_seconds = getattr(env, 'step_period', 3600)

    def _get_comfort_range(self, sim_time):
        """Return (lower_K, upper_K) for the given simulation time (seconds)."""
        if self.comfort_schedule is None:
            return self.comfort_temp_range
        seconds_per_day = 86400.0
        hour_of_day = (sim_time % seconds_per_day) / 3600.0
        for start_h, end_h, bounds in self.comfort_schedule:
            lower, upper = float(bounds[0]), float(bounds[1])
            if start_h < end_h:
                if start_h <= hour_of_day < end_h:
                    return (lower, upper)
            else:
                if hour_of_day >= start_h or hour_of_day < end_h:
                    return (lower, upper)
        return self.comfort_temp_range

    def _get_zone_temp(self, measurements):
        """Return zone temperature (K) from a measurements payload."""
        for var in ('reaTZon_y', 'reaTRooAir_y', 'TRooAir_y', 'TZon_y'):
            val = measurements.get(var)
            if val is None:
                continue
            if isinstance(val, dict):
                v = val.get('value')
                if v is not None:
                    return float(v)
            else:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    pass
        return 293.15   # fallback: ~20 °C

    def _potential(self, zone_temp, sim_time):
        """Φ(s) = −(T_zone − T_center)²  where T_center = midpoint of comfort band."""
        lower, upper = self._get_comfort_range(sim_time)
        t_center = (lower + upper) / 2.0
        return -(zone_temp - t_center) ** 2

    def reset(self, **kwargs):
        """Reset and cache initial zone temperature, sim_time, and KPI energy."""
        obs, info = self.env.reset(**kwargs)
        _base = self.env
        while hasattr(_base, 'env'):
            _base = _base.env
        base_res  = getattr(_base, 'res', None) or {}
        kpis      = getattr(_base, 'last_kpis', {})
        meas      = getattr(_base, 'last_measurements', {}) or base_res
        self.prev_kpi_energy = kpis.get('ener_tot', 0)
        self.prev_sim_time   = float(extract_value(base_res, 'time', 0))
        zone_temp = self._get_zone_temp(meas)
        if zone_temp == 293.15 and base_res:
            zone_temp = self._get_zone_temp(base_res)
        self.prev_zone_temp = zone_temp
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Navigate to base env for cached measurements and KPIs
        _base = self.env
        while hasattr(_base, 'env'):
            _base = _base.env
        _base_res    = getattr(_base, 'res', None) or {}
        measurements = getattr(_base, 'last_measurements', {})
        kpis         = getattr(_base, 'last_kpis', {})

        # ── Zone temperatures ──────────────────────────────────────────────────
        # s_t  : previous state (stored from last step / reset)
        # s_t+1: new state after the action — from this step's measurements
        zone_temp_t1 = self._get_zone_temp(measurements)
        if zone_temp_t1 == 293.15 and _base_res:
            zone_temp_t1 = self._get_zone_temp(_base_res)
        zone_temp_t = self.prev_zone_temp if self.prev_zone_temp is not None else zone_temp_t1

        # ── Simulation times ───────────────────────────────────────────────────
        sim_time_t1 = float(info.get('time', 0))
        sim_time_t  = self.prev_sim_time if self.prev_sim_time is not None else sim_time_t1

        # ── PBRS shaping term: F = γΦ(s') − Φ(s) ─────────────────────────────
        # Φ(s) = −(T_zone − T_center)²  → 0 at centre, increasingly negative away
        phi_s  = self._potential(zone_temp_t,  sim_time_t)
        phi_s1 = self._potential(zone_temp_t1, sim_time_t1)
        F = self.gamma * phi_s1 - phi_s

        # ── Energy base reward ─────────────────────────────────────────────────
        current_kpi_energy = kpis.get('ener_tot', 0)
        delta_energy = current_kpi_energy - (self.prev_kpi_energy or 0)
        safe_e_ref = max(float(self.E_ref), 1e-9)
        r_base = -self.energy_weight * (delta_energy / safe_e_ref)

        # ── Combined PBRS reward ───────────────────────────────────────────────
        reward = r_base + self.comfort_weight * F

        # ── Update state for next step ─────────────────────────────────────────
        self.prev_zone_temp  = zone_temp_t1
        self.prev_sim_time   = sim_time_t1
        self.prev_kpi_energy = current_kpi_energy

        # ── Diagnostics for logging / visualisation ────────────────────────────
        lower_t1, upper_t1 = self._get_comfort_range(sim_time_t1)
        t_center_t1 = (lower_t1 + upper_t1) / 2.0
        e_c = max(0.0, lower_t1 - zone_temp_t1, zone_temp_t1 - upper_t1)
        thermal_comfort = (e_c == 0.0)

        timestep_hours = self.timestep_seconds / 3600.0
        total_power = (delta_energy * 1000.0) / timestep_hours if timestep_hours > 0 else 0.0

        # Keys consumed by episodic_visualisation.py
        info['comfort_reward']       = float(self.comfort_weight * F)
        info['price_reward']         = float(r_base)
        info['shaping_F']            = float(F)
        info['phi_s']                = float(phi_s)
        info['phi_s1']               = float(phi_s1)
        info['thermal_comfort']      = thermal_comfort
        info['zone_temp']            = zone_temp_t1
        info['total_power']          = total_power
        info['kpi_energy']           = current_kpi_energy
        info['delta_energy']         = delta_energy
        info['comfort_temp_lower_K'] = lower_t1
        info['comfort_temp_upper_K'] = upper_t1
        info['t_center']             = t_center_t1
        info['electricity_price']    = 0.0   # no explicit price term in PBRS
        info['cost_step']            = 0.0

        return obs, reward, terminated, truncated, info


# ── ComputedSetpointObsWrapper ────────────────────────────────────────────────

class ComputedSetpointObsWrapper(gym.Wrapper):
    """Appends computed LowerSetp / UpperSetp forecast values to the observation.

    ``bestest_hydronic_heat_pump`` does not expose LowerSetp / UpperSetp as
    BOPTEST forecast points, so fetching them from the API always fails and
    produces a useless constant fallback.  This wrapper derives the setpoints
    analytically from the known comfort schedule and injects them into the
    observation vector as genuine, time-varying signals.

    For each step in the predictive horizon the lower and upper setpoint
    temperatures (Kelvin) are computed and appended:

        obs_extended = [*inner_obs,
                        LowerSetp[t+0], LowerSetp[t+1], …,   # n_steps values
                        UpperSetp[t+0], UpperSetp[t+1], …]   # n_steps values

    Stack position: immediately above ``BoptestGymEnv``, below
    ``NormalizedObservationWrapper``, so that the setpoints are normalised on
    the same footing as all other observations (bounds = SETP_BOUNDS).
    """

    # Bounds for observation-space normalization (same as the old observations_dict entry).
    SETP_BOUNDS = (280., 310.)   # Kelvin

    # Default schedule — matches CustomRewardWrapper.DEFAULT_COMFORT_SCHEDULE.
    DEFAULT_COMFORT_SCHEDULE = [
        (7,  22, (294.15, 297.15)),   # day   07:00–22:00 → 21–24 °C
        (22,  7, (293.15, 295.15)),   # night 22:00–07:00 → 20–22 °C
    ]

    def __init__(
        self,
        env,
        comfort_schedule=None,
        comfort_temp_range=(295.15, 297.15),
        predictive_period=2 * 3600,
        step_period=3600,
        predictive_step=None,
    ):
        super().__init__(env)
        self.comfort_schedule = (
            comfort_schedule
            if comfort_schedule is not None
            else self.DEFAULT_COMFORT_SCHEDULE
        )
        self.comfort_temp_range = comfort_temp_range
        self.step_period = step_period
        self.predictive_step = predictive_step if predictive_step is not None else step_period
        self.n_steps = max(1, int(predictive_period / self.predictive_step))

        lo, hi = self.SETP_BOUNDS
        extra_low  = np.full(2 * self.n_steps, lo,  dtype=np.float32)
        extra_high = np.full(2 * self.n_steps, hi, dtype=np.float32)
        inner = env.observation_space
        self.observation_space = spaces.Box(
            low=np.concatenate([inner.low,  extra_low]),
            high=np.concatenate([inner.high, extra_high]),
            dtype=np.float32,
        )

    def _get_comfort_range(self, sim_time):
        """Return (lower_K, upper_K) for the given simulation time (seconds)."""
        if self.comfort_schedule is None:
            return self.comfort_temp_range
        seconds_per_day = 86400.0
        hour_of_day = (sim_time % seconds_per_day) / 3600.0
        for start_h, end_h, bounds in self.comfort_schedule:
            lower, upper = float(bounds[0]), float(bounds[1])
            if start_h < end_h:
                if start_h <= hour_of_day < end_h:
                    return (lower, upper)
            else:                                        # wraps around midnight
                if hour_of_day >= start_h or hour_of_day < end_h:
                    return (lower, upper)
        return self.comfort_temp_range

    def _append_setpoints(self, obs, sim_time):
        """Compute scheduled setpoints for n_steps ahead and concatenate."""
        lower_vals, upper_vals = [], []
        for i in range(self.n_steps):
            lo, hi = self._get_comfort_range(sim_time + i * self.predictive_step)
            lower_vals.append(lo)
            upper_vals.append(hi)
        extras = np.array(lower_vals + upper_vals, dtype=np.float32)
        return np.concatenate([obs, extras])

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        sim_time = float(info.get('time', 0))
        return self._append_setpoints(obs, sim_time), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # BoptestGymEnv.reset() returns empty info {}; read simulation time
        # from base env's .res (the /initialize response, always value-format).
        base_res = getattr(self.env, 'res', None) or {}
        sim_time = float(extract_value(base_res, 'time', 0))
        return self._append_setpoints(obs, sim_time), info


class LastActionWrapper(gym.Wrapper):
    """Appends the last action taken to the observation vector.

    Gives the agent memory of its previous control decision, which is useful
    for reasoning about thermal inertia (e.g. "I heated last step, so the
    zone temperature may still be rising").

    The appended slot uses the same bounds as the action space, so it is
    naturally in [0, 1] for the heat pump modulation signal and compatible
    with the normalised observation space that precedes it in the stack.

    Stack position: after ``CustomRewardWrapper``, before ``Monitor``.
    """

    def __init__(self, env):
        super().__init__(env)
        n_act = env.action_space.shape[0]
        inner = env.observation_space
        self.observation_space = spaces.Box(
            low=np.concatenate([inner.low,  env.action_space.low]),
            high=np.concatenate([inner.high, env.action_space.high]),
            dtype=np.float32,
        )
        self._last_action = np.zeros(n_act, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_action = np.zeros(self.env.action_space.shape[0], dtype=np.float32)
        return np.concatenate([obs, self._last_action]).astype(np.float32), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_action = np.array(action, dtype=np.float32).flatten()
        return np.concatenate([obs, self._last_action]).astype(np.float32), reward, terminated, truncated, info


class ZoneTempDeltaWrapper(gym.Wrapper):
    """Appends zone temperature rate-of-change (K/step) to the observation.

    Placed between ``ComputedSetpointObsWrapper`` and
    ``NormalizedObservationWrapper`` so the delta is normalised together
    with all other raw observations.

    At this level in the stack ``obs[0]`` is always ``reaTZon_y`` in
    Kelvin (280–310 K) — the raw, un-normalised zone temperature.
    The delta is clipped to ``[DELTA_LOW, DELTA_HIGH]`` K/step before
    being appended; at episode start (reset) the delta is 0.

    Stack position: after ``ComputedSetpointObsWrapper``,
                    before ``NormalizedObservationWrapper``.
    """

    DELTA_LOW  = -5.0   # K/step  (generous lower bound)
    DELTA_HIGH = +5.0   # K/step  (generous upper bound)

    def __init__(self, env):
        super().__init__(env)
        inner = env.observation_space
        self.observation_space = spaces.Box(
            low=np.append(inner.low,   self.DELTA_LOW).astype(np.float32),
            high=np.append(inner.high, self.DELTA_HIGH).astype(np.float32),
            dtype=np.float32,
        )
        self._prev_zone_temp = None   # K, initialised in reset()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_zone_temp = float(obs[0])   # obs[0] = reaTZon_y (K)
        # Delta is 0 at the start of every episode
        return np.append(obs, 0.0).astype(np.float32), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        zone_temp = float(obs[0])
        delta = zone_temp - self._prev_zone_temp if self._prev_zone_temp is not None else 0.0
        delta = float(np.clip(delta, self.DELTA_LOW, self.DELTA_HIGH))
        self._prev_zone_temp = zone_temp
        return np.append(obs, delta).astype(np.float32), reward, terminated, truncated, info


# ── SinCosTimeWrapper ─────────────────────────────────────────────────────────

class SinCosTimeWrapper(gym.Wrapper):
    """Appends a circular (sin, cos) encoding of the current time-of-day.

    Raw time has a hard discontinuity at midnight (86 399 s → 0 s) which
    breaks the smoothness assumption of MLP policies.  The two-component
    circular encoding

        sin(2π × t / 86 400)   ∈ [−1, +1]
        cos(2π × t / 86 400)   ∈ [−1, +1]

    is continuous and periodic everywhere.  Together, sin and cos uniquely
    identify the hour of day (sin alone is ambiguous for AM/PM).

    Two values replace five raw-time slots (4 forecast steps + 1 duplicate
    current measurement), reducing the observation space by 3 dimensions.

    Stack position: after ``ZoneTempDeltaWrapper``,
                    before ``NormalizedObservationWrapper``.
    The NormalizedObservationWrapper maps [−1, +1] → [0, 1] via the declared
    bounds, so no special handling is required downstream.
    """

    _PERIOD = 86400.0   # seconds per day

    def __init__(self, env):
        super().__init__(env)
        inner = env.observation_space
        extra_low  = np.array([-1.0, -1.0], dtype=np.float32)
        extra_high = np.array([+1.0, +1.0], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.concatenate([inner.low,  extra_low]),
            high=np.concatenate([inner.high, extra_high]),
            dtype=np.float32,
        )

    def _encode(self, sim_time):
        """Return (sin, cos) pair for the given simulation time (seconds)."""
        angle = 2.0 * np.pi * (float(sim_time) % self._PERIOD) / self._PERIOD
        return np.array([np.sin(angle), np.cos(angle)], dtype=np.float32)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        sim_time = float(info.get('time', 0))
        return np.concatenate([obs, self._encode(sim_time)]), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Navigate to BoptestGymEnv to read sim time from the initialise response.
        _e = self.env
        while hasattr(_e, 'env'):
            _e = _e.env
        sim_time = float(extract_value(getattr(_e, 'res', None) or {}, 'time', 0))
        return np.concatenate([obs, self._encode(sim_time)]), info

