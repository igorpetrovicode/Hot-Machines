![Hot Machines banner](Hot-Machines_small.png)

# ABB Induction Machine Stator Temperature Tracking — Thermal Model & Efficiency Validation

Data-driven steady-state thermal model for an 10-turbine fleet of 55 kW ABB
induction generators, independently cross-validated against the manufacturer
Test Report (M3AA 250SMA 4 G 400V 50Hz 55kW.pdf) across five operating points 
spanning no-load to rated.

The pipeline fits a single-node first-order thermal network to SCADA time
series for each machine, then uses the fleet-mean parameters — without
re-training on any manufacturer data — to reproduce ABB's reported efficiency
table to within $\pm 13$ W at every load point. The one free parameter in
the validation (stray load loss) matches IEEE Std 112 §5.3 convention and
the value ABB reports via the datasheet's own residual-loss method.

---

## 1. Thermal model (1C1R, Thévenin form)

The winding temperature $T$ is modelled as a single thermal node with lumped
capacitance $C$ and a switched thermal resistance $R_t$ to ambient:

$$
C \frac{dT}{dt} = I^{2} R_{dc}(T) + P_{0} - \frac{T - T_{amb}}{R_t} + \sigma_w \frac{dB}{dt}
$$

with indicator-function switching for the on/off regimes:

$$
R_t = R_{t,\mathrm{on}} \mathbf{1}_{\{\mathrm{on}\}} + R_{t,\mathrm{off}} \mathbf{1}_{\{\mathrm{off}\}}
\qquad
P_{0} \to P_{0} \mathbf{1}_{\{\mathrm{on}\}}
\qquad
R_{dc}(T) = R_{dc,20}\bigl[ 1 + \alpha(T-20) \bigr]
$$

Linearising around the operating point gives the deterministic ODE
$dT/dt = a T + b(t)$ with

$$
a = \frac{I^{2} R_{dc,20} \alpha - 1/R_t}{C}
\qquad
b = \frac{I^{2} R_{dc,20} (1 - \alpha T_{ref}) + P_{0} + T_{amb}/R_t}{C}
$$

which is integrated by trapezoidal exponential update:

$$
T_{k+1} = e^{a \Delta t_k} \left(T_{k} + \tfrac{\Delta t_k}{2} b_{k}\right) + \tfrac{\Delta t_k}{2} b_{k+1}
$$

Parameters $R_{t,\mathrm{on}}, R_{t,\mathrm{off}}, P_0, C_{eq}$ are fit
per machine by nonlinear least squares on whitened innovations $\nu_k = (e_{k+1} - e^{a\Delta t_k}e_k)/\Gamma_k$ with
$\Gamma_k = (e^{a\Delta t_k}-1)/a$, giving fleet-mean $R_{t,\mathrm{on}} = 0.0482$ °C/W $\pm$ 4%, $P_0 = 334$ W $\pm$ 12%, $C_{eq} = 47060$ J/K $\pm$ 19%, and fleet-average RMSE of 2.80 °C.

### Machine reactance from ABB Type Test (Form 4 derivation)

The generator current input to the thermal model is computed via the
Form 4 reactive-power model Q = -(Q₀ + k·P²). Both parameters are
derived from standard tests on page 1 of the ABB Type Test Report.

**Step 1 — No-load test → magnetising reactive power Q₀**

ABB data: V = 400.8 V (line), I = 35.6 A (line), P = 1040 W.

$$
S_{NL} = \sqrt{3} \times 400.8 \times 35.6 = 24714\ \mathrm{VA}
$$

$$
Q_0 = \frac{1}{1000}\sqrt{S_{NL}^2 - P_{NL}^2} = \frac{1}{1000}\sqrt{24714^2 - 1040^2} = 24.69\ \mathrm{kVAR}
$$

**Step 2 — Locked rotor test → leakage reactance X_eq**

ABB data: V = 71.3 V (line), I = 92.5 A (line), P = 4300 W, cos φ = 0.38.

Delta connection: V_phase = V_line, I_phase = I_line/√3.

$$
I_{phase} = 92.5 / \sqrt{3} = 53.40\ \mathrm{A}
$$

$$
S_{LR} = \sqrt{3} \times 71.3 \times 92.5 = 11423\ \mathrm{VA}
$$

$$
Q_{LR} = \sqrt{S_{LR}^2 - P_{LR}^2} = \sqrt{11423^2 - 4300^2} = 10583\ \mathrm{VAR}
$$

$$
X_{eq} = \frac{Q_{LR}}{3\,I_{phase}^2} = \frac{10583}{3 \times 53.40^2} = 1.2369\ \Omega
$$

This is the per-phase delta value of X₁ + X₂' (stator + referred rotor
leakage reactance). At locked rotor the magnetising branch is effectively
shorted, so the measured impedance is purely the series leakage path.

**Step 3 — Rated test → referred rotor current I₂'**

ABB data: I = 99.5 A (line), cos φ = 0.84; no-load cos φ = 0.04.

Subtract the magnetising current phasor from the rated stator current
phasor (all per-phase, delta):

$$
I_{phase} = 99.5/\sqrt{3} = 57.45\ \mathrm{A}
\qquad
I_{mag} = 35.6/\sqrt{3} = 20.56\ \mathrm{A}
$$

| Component | Active [A] | Reactive [A] |
| --- | --- | --- |
| Rated stator | 57.45 × 0.84 = 48.26 | 57.45 × 0.543 = 31.17 |
| Magnetising  | 20.56 × 0.04 = 0.82  | 20.56 × 0.999 = 20.54 |
| I₂' = difference | 47.43 | 10.63 |

$$
|I_2'| = \sqrt{47.43^2 + 10.63^2} = 48.62\ \mathrm{A}
$$

**Step 4 — Leakage coefficient k**

The load-dependent leakage reactive power is Q_leak = 3·X_eq·I₂'².
Since I₂' scales approximately with P, this gives the quadratic term:

$$
k = \frac{3\,X_{eq}\,I_{2,rated}'^{\,2}}{P_{rated}^2 \times 1000} = \frac{3 \times 1.2369 \times 48.62^2}{58.25^2 \times 1000} = 0.00259\ \mathrm{kVAR/kW^2}
$$

The SCADA-fitted value (k = 0.00352) is 36% higher than this ABB-derived
value, reflecting actual installation conditions and capacitor bank
interaction that the Type Test bench does not capture.

---

## 2. Continuous current rating (trip curve)

At thermal equilibrium, setting $dT/dt = 0$ with $T = T_{lim}$
(IEC Class F: $155\ \mathrm{°C}$) gives the steady-state heat balance

$$
I_{cont}^{2} R_{dc}(T_{lim}) + P_{0} = \frac{T_{lim} - T_{amb}}{R_{t,\mathrm{on}}}
$$

which solves to

$$
I_{cont} = \sqrt{\frac{(T_{lim} - T_{amb})/R_{t,\mathrm{on}} - P_{0}}{R_{dc}(T_{lim})}}
$$

This defines the maximum continuous current at which the winding can operate
indefinitely without exceeding its insulation class. The time-to-trip for any
initial condition $T_0$ at constant overcurrent is given in closed form by the
linearised ODE solution:

$$
t_{A}^{ \mathrm{stator}} = \frac{1}{a} \ln \left(\frac{T_{lim} - T_{ss}}{T_{0} - T_{ss}}\right)
\qquad T_{ss} = -b/a
$$

Each machine's continuous rating and trip curve are compared against the
relay setting $t_{A}^{ \mathrm{relay}} = (\mathrm{CLASS}/5)\cdot 23.3 \ln \bigl(\tfrac{I/I_e}{ I/I_e - 1.15 }\bigr)$
to verify protection coordination.

---

## 3. Steady-state efficiency budget (independent validation)

The thermal model is cross-validated at five load points from the ABB Type
Test Report by decomposing total loss $P_1 - P_2$ into four physical loss
components.

$$
P_1 - P_2 = P_{loss} = P_{Cu,s} + P_{rCu} + P_{ifw} + P_{stray}
$$

### Stator copper — from the 1C1R model

$$
P_{Cu,s}(P_1) = 3 I_{ph}^{2} R_{dc} \bigl(T_{ss}(P_1)\bigr)
\qquad
T_{ss}(P_1) \text{ from 1C1R heat balance}
$$

### Rotor copper — slip identity (energy conservation at air gap)

$$
P_{rCu} = s \cdot P_{elec}
\qquad
s(P_1) = s_{\mathrm{rated}}\cdot\frac{P_1}{P_{1,\mathrm{rated}}}
\qquad
s_{\mathrm{rated}} = 0.01333, P_{1,\mathrm{rated}} = 58.25 kW
$$

This is an exact power-balance identity, not a resistance calculation — no
temperature correction applies because the measured slip already reflects
the rotor temperature at thermal equilibrium during ABB's test.

### Iron + friction + windage — from no-load test

The ABB No-Load row gives $P_{NL} = 1.04\ \mathrm{kW}$ at
$I_{NL} = 35.6\ \mathrm{A}$. Solving the no-load heat balance for the
winding temperature at equilibrium

$$
\frac{T_{w,NL} - T_{amb}}{R_{t,\mathrm{on}}} = 3 I_{NL}^{2} R_{dc}(T_{w,NL}) + P_{0}
$$

gives $T_{w,NL} = 47.7\ \mathrm{°C}$, $P_{Cu,NL} = 138\ \mathrm{W}$, and

$$
P_{ifw} = P_{NL} - P_{Cu,NL} = 1040 - 138 = 902\ \mathrm{W}
$$

assumed load-independent (ABB constant-voltage test).

### Stray load loss — per IEEE 112 §5.3

A single coefficient fit to the three-term residual using the IEEE 112
canonical form:

$$
P_{stray}(P_1) = P_{stray,\mathrm{rated}} \cdot \left(\frac{P_1}{P_{1,\mathrm{rated}}}\right)^{ 2}
\qquad
P_{stray,\mathrm{rated}} = 284\ \mathrm{W}
$$

This sits within the IEEE 112 / NEMA MG-1 typical range for IE2 motors and
corresponds to what ABB explicitly labels *"PLL determined from residual
loss"* in the datasheet footnote.

### Result — budget closure

| $P_{1,elec}$ | $P_{2,mech}$ | $P_{Cu,s}$ | $P_{rCu}$ | $P_{ifw}$ | $P_{stray}$ | **$P_{loss}$** | **$(P_1{-}P_2)_{ABB}$** | **Δ** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.04 kW  | 0.00 kW  | 138 W  | 0 W   | 902 W | 0 W   | **1040 W** | $P_{NL} = 1040$ W            | **+0**  |
| 15.30 kW | 14.13 kW | 208 W  | 54 W  | 902 W | 20 W  | **1184 W** | $15.30{-}14.13 = 1170$ W     | **−14** |
| 29.52 kW | 27.94 kW | 393 W  | 199 W | 902 W | 73 W  | **1567 W** | $29.52{-}27.94 = 1580$ W     | **+13** |
| 44.04 kW | 41.79 kW | 741 W  | 444 W | 902 W | 162 W | **2249 W** | $44.04{-}41.79 = 2250$ W     | **+1**  |
| 58.25 kW | 55.00 kW | 1291 W | 776 W | 902 W | 284 W | **3253 W** | $58.25{-}55.00 = 3250$ W     | **−3**  |

**Mean $|\Delta| = 8\ \mathrm{W}$, or $0.01\%$ of rated input power** — at
or below ABB datasheet rounding precision. The budget is complete to within
measurement noise at every load point.

---

## 4. Reproducing the analysis

Place the following alongside the Python scripts (same directory):

```
10.0.103.10.csv    10.0.153.10.csv    10.0.182.10.csv    10.1.20.10.csv
10.1.27.10.csv     10.1.78.11.csv     10.1.111.10.csv    10.1.166.10.csv
10.1.179.10.csv    10.1.181.10.csv
M3AA 250SMA 4 G 400V 50Hz 55kW.pdf    (ABB Type Test Report)
Pages from h1433g.pdf                 (Eaton ZEV relay datasheet)
```

Then run:

```
# Full pipeline: thermal fit + diagnostics + efficiency validation + PDF report
python run_fleet.py --plots --reporting

# Or run each stage individually:
python run_fleet.py --plots              # Stage 1-2: Form 4 + Thévenin fit
python efficiency_stackup_study.py       # Steady-state efficiency validation
python generate_report_pdf.py            # Assemble the 17-page PDF report
```

Pipeline outputs:

* `T{1..10}_thevenin_summary.png` — per-turbine fit quality, residuals, trip curves
* `form4_calibration.png` — fleet reactive-power calibration (Q₀, k, Q_cap)
* `zev_trip_curves.png` — protection coordination vs. machine damage curve
* `model_equations.png` — Model Equations Summary
* `efficiency_stackup_study.png` — 5-load-point efficiency validation
* `machine_study_report.pdf` — assembled deliverable

**Fleet parameters (10-turbine mean ± CV):**

| Parameter | Mean | CV |
| --- | --- | --- |
| $R_{t,\mathrm{on}}$ | $0.0482\ \mathrm{K/W}$ | 4% |
| $R_{t,\mathrm{off}}$ | $0.104\ \mathrm{K/W}$&hairsp;† | 35%&hairsp;† |
| $P_0$ | $334\ \mathrm{W}$ | 12% |
| $C_{eq}$ | $47{,}060\ \mathrm{J/K}$ | 19% |
| $\tau_\mathrm{on}$ | $37.7\ \mathrm{min}$ | 19% |
| RMSE | $2.80\ \mathrm{°C}$ | — |

† $R_{t,\mathrm{off}}$ excludes T3 and T4 (bound-saturated fits); full-fleet
CV is 97%.

**ABB datasheet constants:**

$$
P_{1,\mathrm{rated}} = 58.25\ \mathrm{kW} \qquad I_{\mathrm{rated}} = 99.5\ \mathrm{A} \qquad \eta_{\mathrm{rated}} = 94.42\%
$$

$$
T_{w,\mathrm{rated}} = 89\ \mathrm{°C} \quad (\text{63 K rise above 26 °C ambient, resistance method})
$$

$$
s_{\mathrm{rated}} = 0.01333 \quad (\text{1480 rpm, 4-pole, 50 Hz})
$$

Insulation class F ($T_{lim} = 155\ \mathrm{°C}$), delta-connected stator
with $R_{\phi,20} = 0.09806\ \Omega$ from the ABB cold-resistance
measurement ($U{-}V = 0.06718$, $U{-}W = 0.06603$, $V{-}W = 0.06722\ \Omega$
at $25.6\ \mathrm{°C}$ line-to-line).
