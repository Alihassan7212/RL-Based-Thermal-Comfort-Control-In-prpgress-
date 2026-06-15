# Pseudo-Code: Reinforcement Learning Based HVAC Control
### BOPTEST Environment and SAC Training Pipeline

---

## Algorithm 1: BOPTEST Environment Initialisation

**Input:** URL $u$, testcase name, action variables $A$, observation variables $O$,
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;step period $\Delta t = 1800$ s, episode length $T_{ep} = 259{,}200$ s, warmup $T_w = 86{,}400$ s,
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;predictive horizon $H = 14{,}400$ s, predictive step $h = 3{,}600$ s

**Output:** Initialised environment with action space $A$ and observation space $S$

1: POST $u$/testcases/bestest\_hydronic\_heat\_pump/select $\rightarrow$ testid
2: GET $u$/measurements/testid $\rightarrow$ all\_measurement\_vars
3: GET $u$/forecast\_points/testid $\rightarrow$ all\_forecast\_vars
4: GET $u$/inputs/testid $\rightarrow$ all\_input\_vars
5: PUT $u$/step/testid $\leftarrow \{\Delta t = 1800\}$
6: PUT $u$/scenario/testid $\leftarrow \{$electricity\_price: highly\_dynamic$\}$
7: Define action space $A = [0, 1]$ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\triangleright$ heat pump modulation signal
8: Define observation space $S \subset \mathbf{R}^{21}$ from bounds in $O$ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\triangleright$ raw Kelvin / W/m²
9: Initialise episode\_reward $\leftarrow 0$, objective\_integrand $\leftarrow 0$

---

## Algorithm 2: Episode Reset

**Input:** Random start flag, excluding periods $E_{excl}$, warmup period $T_w$

**Output:** Initial observation $s_0 \in S$

1: **if** random\_start\_time **then**
2: &nbsp;&nbsp;&nbsp;&nbsp; Repeat: $t_0 \sim \text{Uniform}(0,\ T_{year} - T_{ep})$
3: &nbsp;&nbsp;&nbsp;&nbsp; **until** $[t_0,\ t_0 + T_{ep}]$ does not overlap any period in $E_{excl}$
4: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\triangleright$ exposes agent to all seasons for generalisation
5: **end if**
6: PUT $u$/initialize/testid $\leftarrow \{$start\_time: $t_0$, warmup\_period: $T_w\}$
7: GET $u$/kpi/testid $\rightarrow$ kpis
8: objective\_integrand $\leftarrow$ kpis[cost\_tot] $+$ kpis[tdis\_tot]
9: $s_0 \leftarrow$ **GetObservation**() &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\triangleright$ Algorithm 3
10: **return** $s_0$

---

## Algorithm 3: Observation Construction

**Input:** Current simulation response res, simulation time $t$, predictive horizon $H$, step $h$

**Output:** Raw observation vector $\mathbf{o} \in \mathbf{R}^{17}$ (before wrapper augmentation)

1: measurements $\leftarrow$ res &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\triangleright$ reuse POST /advance/ response; avoids redundant HTTP call
2: PUT $u$/forecast/testid $\leftarrow \{$point\_names: [TDryBul, HDirNor], horizon: $H$, interval: $h\}$
3: forecast\_data $\leftarrow$ response &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\triangleright$ 4-step horizon: $t$+0, $t$+60, $t$+120, $t$+180 min
4: $\mathbf{o} \leftarrow [\ ]$
5: **for** each obs\_name in $\{$reaTZon\_y, TDryBul, HDirNor$\}$ **do**
6: &nbsp;&nbsp;&nbsp;&nbsp; **if** obs\_name is a forecast variable **then**
7: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **for** $i = 0$ **to** $n_{steps} - 1$ **do** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\triangleright$ $n_{steps} = H / h = 4$
8: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Append forecast\_data[obs\_name][$i$] to $\mathbf{o}$
9: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **end for**
10: &nbsp;&nbsp;&nbsp;&nbsp; **end if**
11: &nbsp;&nbsp;&nbsp;&nbsp; **if** obs\_name is a measurement variable **then**
12: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Append measurements[obs\_name] to $\mathbf{o}$
13: &nbsp;&nbsp;&nbsp;&nbsp; **end if**
14: **end for**
15: **return** $\mathbf{o}$ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\triangleright$ shape: (1 + 4 + 4) = 9 dims; extended to 21 by wrappers below

---

## Algorithm 4: Observation Wrapper Stack

**Input:** Raw observation $\mathbf{o} \in \mathbf{R}^{9}$, simulation time $t$, comfort schedule $C$,
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;previous zone temperature $T^{prev}_{zone}$, previous action $a_{t-1}$

**Output:** Normalised, augmented observation $\mathbf{s} \in [0,1]^{21}$

1: $\triangleright$ **ComputedSetpointObsWrapper** — inject LowerSetp / UpperSetp forecast
2: **for** $i = 0$ **to** $3$ **do**
3: &nbsp;&nbsp;&nbsp;&nbsp; $(T^{lower}_i,\ T^{upper}_i) \leftarrow$ GetComfortRange$(t + i \cdot h,\ C)$
4: **end for**
5: $\mathbf{o} \leftarrow [\mathbf{o},\ T^{lower}_0, \ldots, T^{lower}_3,\ T^{upper}_0, \ldots, T^{upper}_3]$ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\triangleright$ +8 dims → 17 dims
6:
7: $\triangleright$ **ZoneTempDeltaWrapper** — thermal inertia signal
8: $\Delta T_{zone} \leftarrow \text{clip}(T_{zone} - T^{prev}_{zone},\ -5,\ +5)$ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\triangleright$ K/step
9: $\mathbf{o} \leftarrow [\mathbf{o},\ \Delta T_{zone}]$ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\triangleright$ +1 dim → 18 dims
10:
11: $\triangleright$ **SinCosTimeWrapper** — circular time-of-day encoding; eliminates midnight discontinuity
12: $\mathbf{o} \leftarrow \left[\mathbf{o},\ \sin\!\left(\tfrac{2\pi\, (t \bmod 86400)}{86400}\right),\ \cos\!\left(\tfrac{2\pi\, (t \bmod 86400)}{86400}\right)\right]$ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\triangleright$ +2 dims → 20 dims
13:
14: $\triangleright$ **NormalizedObservationWrapper** — min-max normalise all features
15: $\mathbf{o} \leftarrow \text{clip}\!\left(\dfrac{\mathbf{o} - \mathbf{o}^{low}}{\mathbf{o}^{high} - \mathbf{o}^{low}},\ 0,\ 1\right)$
16:
17: $\triangleright$ **LastActionWrapper** — memory of previous decision
18: $\mathbf{s} \leftarrow [\mathbf{o},\ a_{t-1}]$ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\triangleright$ +1 dim → 21 dims, normalised
19: **return** $\mathbf{s}$

---

## Algorithm 5: Hat-Shaped Reward Function

**Input:** Zone temperature $T_{zone}$, comfort bounds $T_{lower}$, $T_{upper}$,
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;energy consumed $E$ (kWh/m²), electricity price $p(t)$, slope $k$, energy weight $\omega$

**Output:** Step reward $r$

1: **if** $T_{lower} \leq T_{zone} \leq T_{upper}$ **then**
2: &nbsp;&nbsp;&nbsp;&nbsp; $r_{comfort} \leftarrow +1.0$
3: **else if** $T_{zone} < T_{lower}$ **then**
4: &nbsp;&nbsp;&nbsp;&nbsp; $r_{comfort} \leftarrow -k \cdot (T_{lower} - T_{zone})$
5: **else**
6: &nbsp;&nbsp;&nbsp;&nbsp; $r_{comfort} \leftarrow -k \cdot (T_{zone} - T_{upper})$
7: **end if**
8: $r_{energy} \leftarrow -\omega \cdot E \cdot p(t)$
9: $r \leftarrow r_{comfort} + r_{energy}$
10: **return** $r$

---

## Algorithm 6: Huber-Band Reward Function

**Input:** Zone temperature $T_{zone}$, comfort bounds $T_{lower}$, $T_{upper}$,
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;energy consumed $E$ (kWh/m²), electricity price $p(t)$,
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;comfort weight $w_c = 50$, price weight $w_p = 30$, Huber threshold $\delta = 0.25$ K

**Output:** Step reward $r$

1: $e_c \leftarrow \max(0,\ T_{lower} - T_{zone},\ T_{zone} - T_{upper})$ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\triangleright$ violation magnitude in K
2: **if** $e_c \leq \delta$ **then**
3: &nbsp;&nbsp;&nbsp;&nbsp; $r_{comfort} \leftarrow -w_c \cdot \dfrac{e_c^2}{2\delta}$ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\triangleright$ quadratic near band edge
4: **else**
5: &nbsp;&nbsp;&nbsp;&nbsp; $r_{comfort} \leftarrow -w_c \cdot \left(e_c - \dfrac{\delta}{2}\right)$ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\triangleright$ linear for large violations
6: **end if**
7: $r_{energy} \leftarrow -w_p \cdot E \cdot p(t)$
8: $r \leftarrow r_{comfort} + r_{energy}$
9: **return** $r$

---

## Algorithm 7: Potential-Based Reward Shaping (PBRS)

**Input:** Previous state $(T^t_{zone},\ t_{sim}^t)$, next state $(T^{t+1}_{zone},\ t_{sim}^{t+1})$,
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;energy delta $\Delta E$ (kWh/m²), electricity price $p(t)$, comfort schedule $C$,
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\gamma = 0.995$, $w_c = 10$, $w_e = 1$, $E_{ref} = 0.01$ kWh/m²/step

**Output:** Step reward $r$

1: $(T^t_{lower},\ T^t_{upper}) \leftarrow$ GetComfortRange$(t_{sim}^t,\ C)$
2: $(T^{t+1}_{lower},\ T^{t+1}_{upper}) \leftarrow$ GetComfortRange$(t_{sim}^{t+1},\ C)$
3: $T^t_{center} \leftarrow \dfrac{T^t_{lower} + T^t_{upper}}{2}$, &nbsp;&nbsp; $T^{t+1}_{center} \leftarrow \dfrac{T^{t+1}_{lower} + T^{t+1}_{upper}}{2}$
4: $\Phi(s) \leftarrow -(T^t_{zone} - T^t_{center})^2$ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\triangleright$ higher potential = closer to band centre
5: $\Phi(s') \leftarrow -(T^{t+1}_{zone} - T^{t+1}_{center})^2$
6: $F \leftarrow \gamma \cdot \Phi(s') - \Phi(s)$ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\triangleright$ policy-invariant shaping term (Ng et al., 1999)
7: $r_{base} \leftarrow -w_e \cdot \dfrac{\Delta E \cdot p(t)}{E_{ref}}$ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\triangleright$ normalised energy penalty
8: $r \leftarrow r_{base} + w_c \cdot F$
9: **return** $r$

---

## Algorithm 8: SAC Training with Parallel BOPTEST Environments

**Input:** Number of parallel environments $N = 8$, total vectorised steps $M = 80{,}000$,
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;learning rate $\alpha = 3 \times 10^{-4}$, discount factor $\gamma = 0.995$,
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;replay buffer size $|B| = 100{,}000$, batch size $b = 256$

**Output:** Trained SAC policy $\pi_\theta$, KPI history

1: **for** $i = 1$ **to** $N$ **do**
2: &nbsp;&nbsp;&nbsp;&nbsp; $\text{env}_i \leftarrow$ CreateEnv(subprocess=True) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\triangleright$ each env owns a separate BOPTEST testid
3: **end for**
4: $V \leftarrow$ SubprocVecEnv$(\text{env}_1, \ldots, \text{env}_N)$ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\triangleright$ true HTTP parallelism across $N$ processes
5: Initialise SAC policy $\pi_\theta$ with MLP $(2 \times 256,\ \text{ReLU})$, entropy $H =$ auto
6: Initialise replay buffer $B \leftarrow \emptyset$
7: $\mathbf{s} \leftarrow V$.reset() &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\triangleright$ $\mathbf{s} \in \mathbf{R}^{N \times 21}$
8: **for** $k = 1$ **to** $M$ **do**
9: &nbsp;&nbsp;&nbsp;&nbsp; $\mathbf{a} \leftarrow \pi_\theta(\mathbf{s})$ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\triangleright$ sample action for all $N$ envs simultaneously
10: &nbsp;&nbsp;&nbsp;&nbsp; $\mathbf{s}', \mathbf{r}, \mathbf{d} \leftarrow V$.step$(\mathbf{a})$ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\triangleright$ $N$ parallel HTTP calls to BOPTEST
11: &nbsp;&nbsp;&nbsp;&nbsp; Store $(\mathbf{s}, \mathbf{a}, \mathbf{r}, \mathbf{s}', \mathbf{d})$ in $B$
12: &nbsp;&nbsp;&nbsp;&nbsp; $\mathbf{s} \leftarrow \mathbf{s}'$
13: &nbsp;&nbsp;&nbsp;&nbsp; **if** $|B| \geq b$ **then**
14: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Sample minibatch $\{(s_j, a_j, r_j, s'_j, d_j)\}_{j=1}^{b}$ from $B$
15: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Update critic: minimise $L_Q(\phi) = \mathrm{E}\!\left[(Q_\phi(s,a) - y)^2\right]$
16: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; where $y = r + \gamma(1-d)\left[Q_{\bar\phi}(s', \tilde a') - \alpha \log \pi_\theta(\tilde a' | s')\right]$
17: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Update actor: maximise $\mathrm{E}\!\left[Q_\phi(s, \tilde a) - \alpha \log \pi_\theta(\tilde a | s)\right]$
18: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Update entropy coefficient $\alpha$: minimise $\mathrm{E}\!\left[-\alpha \log \pi_\theta(a|s) - \alpha H_{target}\right]$
19: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Soft update target network: $\bar\phi \leftarrow \tau\phi + (1-\tau)\bar\phi$
20: &nbsp;&nbsp;&nbsp;&nbsp; **end if**
21: &nbsp;&nbsp;&nbsp;&nbsp; **if** $k \bmod 1000 = 0$ **then**
22: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; kpis $\leftarrow \frac{1}{N}\sum_{i=1}^{N}$ env$_i$.get\_kpis() &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\triangleright$ averaged across all envs
23: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Log: ener\_tot, tdis\_tot, cost\_tot, mean reward per step
24: &nbsp;&nbsp;&nbsp;&nbsp; **end if**
25: **end for**
26: Save $\pi_\theta$ to disk; save config.json, kpi\_history
27: **return** $\pi_\theta$, kpi\_history

---

## Algorithm 9: Agent Evaluation

**Input:** Trained policy $\pi_\theta$, number of episodes $n = 2$, evaluation environment env

**Output:** Per-episode KPIs: energy (kWh/m²), discomfort (K·h/m²), cost (EUR/m²)

1: **for** episode $= 1$ **to** $n$ **do**
2: &nbsp;&nbsp;&nbsp;&nbsp; $s \leftarrow$ env.reset() &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\triangleright$ deterministic start time; same pricing conditions for all controllers
3: &nbsp;&nbsp;&nbsp;&nbsp; done $\leftarrow$ False, &nbsp; $R \leftarrow 0$
4: &nbsp;&nbsp;&nbsp;&nbsp; **while** not done **do**
5: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $a \leftarrow \pi_\theta(s,\ \text{deterministic}=\text{True})$ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $\triangleright$ no exploration noise
6: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $s', r,$ terminated, truncated $\leftarrow$ env.step$(a)$
7: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $R \leftarrow R + r$
8: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; done $\leftarrow$ terminated $\vee$ truncated
9: &nbsp;&nbsp;&nbsp;&nbsp; **end while**
10: &nbsp;&nbsp;&nbsp;&nbsp; kpis $\leftarrow$ env.get\_kpis()
11: &nbsp;&nbsp;&nbsp;&nbsp; Record ener\_tot, tdis\_tot, cost\_tot
12: **end for**
13: **return** mean KPIs across all episodes
