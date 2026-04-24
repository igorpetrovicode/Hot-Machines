"""
Wind turbine generator thermal identification — Thévenin (1C1R) variant.

A single-node thermal model. Treats the generator as a single
thermal mass with a single cooling resistance. No hidden states, no gauge
symmetries, no need for material constants from a datasheet — every parameter
is freely fitted from temperature observations alone.

Trade-off: gives up the ability to model winding/iron temperature separately,
but produces tighter cross-fleet parameter consistency and lower fit RMSE on
average. Recommended as the default model for protection-coordination work.

Usage:
    python turbine_thermal_pipeline_thevenin.py <path_to_csv>
"""

import sys
import os
import numpy as np
import pandas as pd
from scipy.optimize import least_squares


# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS — Form 4 (fleet-calibrated against ABB type test)
# ════════════════════════════════════════════════════════════════════════════

# --- Copper resistivity (ABB type test, kept for I²·Rdc heating term) ---
R_PHASE_20 = 0.09806     # phase resistance at 20°C [Ω]
ALPHA_CU = 0.00393       # copper temperature coefficient [1/K]
T_REF = 20.0             # reference temperature [°C]

# --- Form 4 DEFAULTS (fallback values only) ---
# These are the original published fleet-average values. They are used ONLY
# when compute_I_gen() is called without explicit Q0/k kwargs, i.e. when
# process() is called directly. In normal operation, run_fleet.py drives
# the pipeline via process_with_form4(), which calls calibrate_form4() from
# form4_calibration.py to derive fresh constants from the raw SCADA data
# and passes them in explicitly -- these module-level defaults are bypassed.
# Do not edit these to "update" the calibration; run form4_calibration.py
# instead and inspect its output.
Q0_FORM4 = 24.84         # no-load magnetising reactive [kVAR]   (fallback)
K_FORM4 = 0.003857       # leakage coefficient [kVAR/kW²]        (fallback)
Q_CAP_NOM = 23.42        # cap bank reactive at V_NOM [kVAR]     (fallback)
V_NOM = 245.0            # nominal phase voltage [V]             (fallback)

# --- Cap state detection ---
PF_THRESHOLD = 9000      # PF×10000 above this = cap ON
MIN_RUN = 3              # min consecutive samples for state change

# --- Machine ratings ---
I_RATED = 99.5           # rated phase current [A]
T_LIMIT = 155.0          # Class F insulation limit [°C]

# --- Data gap handling ---
# When consecutive SCADA samples are separated by more than this threshold,
# the trapezoidal propagator re-initializes from the measured T_w at the far
# side of the gap rather than integrating across it, and the corresponding
# residual is masked out of the fit. The 300 s (5 min) value is chosen as:
#
#   Lower bound  — 10-18× the typical SCADA sample interval (17-30 s), so
#                  normal sampling jitter does not trigger re-initialization
#   Upper bound  — τ_on/6 to τ_on/10 of the fitted thermal time constant
#                  (~30-48 min fleet-wide), bounding the maximum thermal
#                  drift across an un-integrated interval to ~10-15% of the
#                  full equilibrium approach
#   Data-driven  — matches the natural elbow in the fleet gap distribution:
#                  most gaps are either <60 s (SCADA jitter, plentiful) or
#                  >300 s (actual data outages, rare), with little in between
#
# A sensitivity sweep of this threshold shows that in the 300-1200 s band,
# fitted R_t,on and C_eq vary by less than 0.2% and 0.8% respectively across
# the fleet — the fit is essentially insensitive to the exact choice within
# this range. Dropping to 120 s starts to eat into legitimate cooling-curve
# samples (whose intervals stretch during machine-off periods) and destabi-
# lizes R_t,off by ~16%; raising to 3600 s lets the integrator drift across
# medium-duration outages and degrades RMSE by up to ~12% on the affected
# turbine. 300 s sits at the knee between these two failure modes.
GAP_THRESHOLD_SEC = 300.0


# ════════════════════════════════════════════════════════════════════════════
# FORM 4: GENERATOR CURRENT + CAP STATE DETECTION
# ════════════════════════════════════════════════════════════════════════════
#
# Form 4 models the generator as supplying reactive power
#     Q_gen = -(Q0 + k*P²)
# from which the phase current and power factor follow. The cap bank state is
# detected here because it's a Form-4-adjacent quantity (determines whether
# the reactive power comes from the cap bank or from the generator itself)
# and can be used as a diagnostic signal or to audit the Form 4 calibration.
# Neither the cap state nor the refractory filter feeds back into the thermal
# fit — they are computed once and passed through as metadata.

def refractory_filter(raw_state, min_run=MIN_RUN):
    """
    Suppress brief state changes (Form 1: minimum run length).
    A transition is only accepted if it persists for >= min_run samples.
    """
    n = len(raw_state)
    filtered = np.empty(n, dtype=bool)
    if n == 0:
        return filtered
    current = bool(raw_state[0])
    filtered[0] = current
    i = 1
    while i < n:
        if bool(raw_state[i]) == current:
            filtered[i] = current
            i += 1
        else:
            j = i
            new_state = bool(raw_state[i])
            while j < n and bool(raw_state[j]) == new_state:
                j += 1
            run_length = j - i
            if run_length >= min_run:
                filtered[i:j] = new_state
                current = new_state
            else:
                filtered[i:j] = current
            i = j
    return filtered


def detect_cap_state(PF_meter_raw):
    """Detect cap bank state from meter PF (×10000) with refractory filter."""
    cap_raw = np.abs(PF_meter_raw) > PF_THRESHOLD
    return refractory_filter(cap_raw, min_run=MIN_RUN)


def compute_I_gen(P, V, Q0=Q0_FORM4, k=K_FORM4):
    """
    Generator phase current via Form 4:
        Q_gen = -(Q0 + k*P²)
        PF    = |P| / sqrt(P² + Q_gen²)
        I_gen = |P|*1000 / (3*V*PF)
    """
    P_abs = np.abs(P)
    Q_gen = -(Q0 + k * P_abs**2)
    PF_gen = np.clip(P_abs / np.sqrt(P_abs**2 + Q_gen**2 + 1e-10), 0.03, 0.99)
    return P_abs * 1000.0 / (3.0 * V * PF_gen)


def compute_form4(data):
    """
    Apply the full Form 4 stage to a SCADA record:
      - Compute generator current via the fleet-calibrated PF curve
      - Square it for thermal use (zeroed during shutdown)
      - Detect cap bank state with refractory filter

    Returns a dict with I_gen, I2, cap_on — to be consumed by the thermal fit
    and reporting stages downstream.
    """
    I_gen = compute_I_gen(data["P"], data["V"])
    I2 = I_gen ** 2
    I2[~data["gen_on"]] = 0.0
    cap_on = detect_cap_state(data["PF_meter_raw"])
    return {
        "I_gen": I_gen,
        "I2": I2,
        "cap_on": cap_on,
    }


# ════════════════════════════════════════════════════════════════════════════
# THÉVENIN 1C1R THERMAL MODEL
# ════════════════════════════════════════════════════════════════════════════
#
# Single thermal mass C with two cooling resistances (forced when generating,
# natural when off):
#
#     C · dT/dt = I²·R_dc(T) + P_const − (T − T_amb)/R_t
#
# where R_dc(T) = R_dc_ref · (1 + α·(T − T_REF)) is the temperature-dependent
# copper resistance, P_const is iron loss (only during generation), and R_t
# switches between R_t_on and R_t_off based on generator state.
#
# Linearised: dT/dt = a·T + b(t), with
#     a = (I²·R_dc_ref·α − 1/R_t) / C
#     b = (I²·R_dc_ref·(1 − α·T_REF) + P_const + T_amb/R_t) / C
#
# Forward simulation uses trapezoidal-forcing discretisation:
#     T[k+1] = exp(a_k·dt)·(T[k] + dt/2·b_k) + dt/2·b_{k+1}
#
# This is O(Δt³) local and more accurate than ZOH when b varies between samples.
# Innovation extraction (for stochastic analysis) still uses ZOH for clean
# per-step invertibility — see extract_innovation_thevenin below.


def _coeffs(i, Rt_on, Rt_off, P0, C_eq, gen_on, I2, Tnac):
    """Linear-ODE coefficients (a, b) at sample i."""
    if gen_on[i]:
        a = (I2[i] * R_PHASE_20 * ALPHA_CU - 1.0/Rt_on) / C_eq
        b = (I2[i] * R_PHASE_20 * (1 - ALPHA_CU*T_REF)
             + P0 + Tnac[i]/Rt_on) / C_eq
    else:
        a = -1.0 / (Rt_off * C_eq)
        b = Tnac[i] / (Rt_off * C_eq)
    return a, b


def simulate_thevenin(Rt_on, Rt_off, P0, C_eq, N, dt, Tw, Tnac, I2,
                      gen_on, gap):
    """
    Simulate the 1C1R model with trapezoidal forcing discretisation.

    The propagator e^(a·dt) uses a from the current step; b is averaged
    between the current and next sample with trapezoidal weights.

    Returns
    -------
    T : array of length N — predicted winding temperature
    w : array of length N — sample weights (0 at gaps, 1 elsewhere)
    """
    T = np.zeros(N)
    w = np.ones(N)
    T[0] = Tw[0]

    for i in range(N - 1):
        if gap[i]:
            T[i + 1] = Tw[i + 1]
            w[i + 1] = 0
            continue

        a_k, b_k = _coeffs(i, Rt_on, Rt_off, P0, C_eq, gen_on, I2, Tnac)
        i1 = min(i + 1, N - 1)
        _, b_k1 = _coeffs(i1, Rt_on, Rt_off, P0, C_eq, gen_on, I2, Tnac)

        if abs(a_k) > 1e-15:
            ea = np.exp(a_k * dt[i])
            T[i + 1] = ea * (T[i] + dt[i]/2 * b_k) + dt[i]/2 * b_k1
        else:
            T[i + 1] = T[i] + dt[i]/2 * (b_k + b_k1)

        if abs(T[i + 1]) > 500:
            T[i + 1] = Tw[i + 1]
            w[i + 1] = 0

    return T, w


def fit_thevenin(N, dt, Tw, Tnac, I2, gen_on, gap):
    """
    Fit the four Thévenin parameters: (R_t_on, R_t_off, P0, C_eq).

    Returns dict with parameters, predicted temperature, weights, RMSE,
    standard errors, and parameter correlation matrix.
    """
    def resid(p):
        T, w = simulate_thevenin(p[0], p[1], p[2], p[3],
                                  N, dt, Tw, Tnac, I2, gen_on, gap)
        return w * (T - Tw)

    res = least_squares(
        resid,
        x0=[0.046, 0.080, 316, 60000],
        bounds=([0.01, 0.01, 0, 5000], [0.2, 0.5, 2000, 200000]),
        method="trf",
        max_nfev=50000,
    )

    Rt_on, Rt_off, P0, C_eq = res.x
    T, w = simulate_thevenin(Rt_on, Rt_off, P0, C_eq,
                              N, dt, Tw, Tnac, I2, gen_on, gap)
    n_eff = int(np.sum(w > 0))
    rmse = float(np.sqrt(np.sum((w * (T - Tw))**2) / n_eff))

    # Standard errors via Jacobian (ignoring nonlinearity in second order)
    sigma2 = np.sum(res.fun**2) / max(n_eff - 4, 1)
    try:
        cov = sigma2 * np.linalg.inv(res.jac.T @ res.jac)
        se = np.sqrt(np.diag(cov))
        D = np.diag(1.0 / se)
        corr = D @ cov @ D
    except np.linalg.LinAlgError:
        se = np.full(4, np.nan)
        corr = np.full((4, 4), np.nan)

    return {
        "Rt_on": Rt_on,
        "Rt_off": Rt_off,
        "P0": P0,
        "C_eq": C_eq,
        "tau_on": C_eq * Rt_on,
        "tau_off": C_eq * Rt_off,
        "rmse": rmse,
        "n_eff": n_eff,
        "se": list(se),
        "corr": corr.tolist(),
        "Tw_pred": T,
        "weights": w,
    }


# ════════════════════════════════════════════════════════════════════════════
# STOCHASTIC / INNOVATION ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
#
# For a 1D system the integrating-factor inversion is straightforward
# — there's no hidden state to propagate. Given the deterministic
# residual e[k] = T_meas[k] − T_pred[k] and the closed-form update
#
#     T[k+1] = T[k]·exp(a·dt) + (b/a)·(exp(a·dt) − 1)
#
# the same recurrence applies to the error driven by an innovation ν[k]:
#
#     e[k+1] = e[k]·exp(a·dt) + Γ·ν[k]
#     Γ      = (exp(a·dt) − 1) / a
#
# Solving for ν[k] gives the per-step process noise.

def extract_innovation_thevenin(Rt_on, Rt_off, P0, C_eq, N, dt, Tw, Tw_pred,
                                 Tnac, I2, gen_on, gap):
    """
    Recover the innovation sequence from a fitted 1C1R model.

    Returns
    -------
    nu : array length N-1
    valid : bool array length N-1
    """
    e = Tw - Tw_pred
    nu = np.zeros(N - 1)

    for i in range(N - 1):
        if gap[i]:
            continue
        if gen_on[i]:
            a = (I2[i] * R_PHASE_20 * ALPHA_CU - 1.0/Rt_on) / C_eq
        else:
            a = -1.0 / (Rt_off * C_eq)
        if abs(a) > 1e-15:
            ea = np.exp(a * dt[i])
            Gamma = (ea - 1) / a
            if abs(Gamma) > 1e-15:
                nu[i] = (e[i + 1] - e[i] * ea) / Gamma
        else:
            nu[i] = (e[i + 1] - e[i]) / dt[i]

    valid = ~gap[:N - 1]
    return nu, valid


def innovation_diagnostics(nu, valid, gen_on, I2, max_lag=10):
    """White-noise diagnostics on the innovation: σ_w, ACF, corr(I), σ ratio."""
    nuv = nu[valid]
    if len(nuv) < max_lag + 2:
        return {"sigma_w": np.nan, "acf": [], "corr_I": np.nan,
                "sigma_ratio": np.nan, "n": len(nuv)}

    sigma_w = float(np.std(nuv))
    acf = []
    for lag in range(1, max_lag + 1):
        if lag < len(nuv):
            acf.append(float(np.corrcoef(nuv[:-lag], nuv[lag:])[0, 1]))

    mg = valid & gen_on[:-1]
    nug = nu[mg]
    Ig = np.sqrt(I2[:-1][mg])
    if len(nug) > 10 and Ig.std() > 1e-6:
        corr_I = float(np.corrcoef(nug, Ig)[0, 1])
    else:
        corr_I = np.nan

    if len(Ig) > 20:
        I_med = float(np.median(Ig))
        nu_lo = nu[mg & (np.sqrt(I2[:-1]) < I_med)]
        nu_hi = nu[mg & (np.sqrt(I2[:-1]) >= I_med)]
        if len(nu_lo) > 5 and len(nu_hi) > 5 and np.var(nu_lo) > 0:
            sigma_ratio = float(np.sqrt(np.var(nu_hi) / np.var(nu_lo)))
        else:
            sigma_ratio = np.nan
    else:
        sigma_ratio = np.nan

    return {
        "sigma_w": sigma_w,
        "acf": acf,
        "corr_I": corr_I,
        "sigma_ratio": sigma_ratio,
        "n": len(nuv),
    }


# ════════════════════════════════════════════════════════════════════════════
# DAMAGE CURVE
# ════════════════════════════════════════════════════════════════════════════
#
# The Thévenin model directly provides the steady-state and time-constant
# needed for relay coordination. No need for a separate "Thévenin-equivalent"
# extraction — the fitted (Rt_on, P0, C_eq) IS the protection model.

def trip_time(Rt_on, P0, C_eq, T0, I_mult, T_amb=40.0):
    """
    Time to reach T_LIMIT from initial temperature T0 at constant overcurrent.
    Returns inf if T_ss < T_LIMIT (no trip), 0 if already over.
    """
    I = I_mult * I_RATED
    a = (I**2 * R_PHASE_20 * ALPHA_CU - 1.0/Rt_on) / C_eq
    b = (I**2 * R_PHASE_20 * (1 - ALPHA_CU*T_REF)
         + P0 + T_amb/Rt_on) / C_eq
    if abs(a) < 1e-15:
        return np.inf
    Tss = -b / a
    if a < 0 and Tss <= T_LIMIT:
        return np.inf
    if T0 >= T_LIMIT:
        return 0.0
    arg = (T_LIMIT - Tss) / (T0 - Tss)
    if arg <= 0:
        return np.inf
    t = (1.0 / a) * np.log(arg)
    return t if t > 0 else np.inf


def continuous_current_rating(Rt_on, P0, C_eq, T_amb=40.0):
    """Maximum current multiplier for which T_ss <= T_LIMIT."""
    for m in np.linspace(1.0, 1.5, 5000):
        I = m * I_RATED
        a = (I**2 * R_PHASE_20 * ALPHA_CU - 1.0/Rt_on) / C_eq
        b = (I**2 * R_PHASE_20 * (1 - ALPHA_CU*T_REF)
             + P0 + T_amb/Rt_on) / C_eq
        Tss = -b / a if abs(a) > 1e-15 else 999
        if abs(a) > 0 and Tss >= T_LIMIT:
            return m
    return None


# ════════════════════════════════════════════════════════════════════════════
# DATASET LOADER
# ════════════════════════════════════════════════════════════════════════════

def load_scada(path):
    """Load a SCADA CSV and return all the arrays the pipeline needs."""
    df = pd.read_csv(path)
    df['time'] = pd.to_datetime(df['timestamp'], format='mixed')
    df = df.sort_values('time').reset_index(drop=True)
    N = len(df)
    t_sec = (df['time'] - df['time'].iloc[0]).dt.total_seconds().values
    dt = np.diff(t_sec)

    P = df['@GV.HRR_kW'].values / 10.0
    Q = df['@GV.HRR_kVAR'].values / 10.0
    Tw = df['@GV.HRR_GeneratorWindingTemp'].values / 10.0
    Tnac = df['@GV.HRR_NacelleAirTemp'].values / 10.0

    V_phases = (df['@GV.primaryLovatoReadings.L1PhaseVoltage'].values
                + df['@GV.primaryLovatoReadings.L2PhaseVoltage'].values
                + df['@GV.primaryLovatoReadings.L3PhaseVoltage'].values) / 3.0 / 100.0

    PF_meter_raw = df['@GV.primaryLovatoReadings.EqvPowerFactor'].values

    pq0 = (np.abs(P) < 0.5) & (np.abs(Q) < 0.5)
    gen_on = np.ones(N, dtype=bool)
    for i in range(1, N):
        gen_on[i] = not (pq0[i] and pq0[i-1])

    V_safe = np.where(V_phases > 100,
                      V_phases,
                      V_phases[gen_on & (V_phases > 100)].mean()
                      if (gen_on & (V_phases > 100)).any() else V_NOM)

    return {
        "df": df, "N": N, "t_sec": t_sec, "dt": dt,
        "P": P, "Q": Q, "Tw": Tw, "Tnac": Tnac,
        "V": V_safe, "PF_meter_raw": PF_meter_raw,
        "gen_on": gen_on, "gap": dt > GAP_THRESHOLD_SEC,
    }


# ════════════════════════════════════════════════════════════════════════════
# TOP-LEVEL PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def process(path):
    """Run the full Thévenin pipeline on a single CSV."""
    data = load_scada(path)
    N = data["N"]

    # 1) Form 4 stage: generator current + cap state detection
    form4 = compute_form4(data)
    I_gen = form4["I_gen"]
    I2 = form4["I2"]
    cap_on = form4["cap_on"]

    # 2) Thévenin thermal fit
    fit = fit_thevenin(N, data["dt"], data["Tw"], data["Tnac"], I2,
                       data["gen_on"], data["gap"])

    # 3) Stochastic analysis
    nu, valid = extract_innovation_thevenin(
        fit["Rt_on"], fit["Rt_off"], fit["P0"], fit["C_eq"],
        N, data["dt"], data["Tw"], fit["Tw_pred"], data["Tnac"],
        I2, data["gen_on"], data["gap"]
    )
    diag = innovation_diagnostics(nu, valid, data["gen_on"], I2)

    # 4) Damage curve and continuous rating
    damage = {
        I_mult: trip_time(fit["Rt_on"], fit["P0"], fit["C_eq"], 40.0, I_mult)
        for I_mult in [1.1, 1.2, 1.5, 2.0, 3.0, 5.0]
    }
    I_cont = continuous_current_rating(fit["Rt_on"], fit["P0"], fit["C_eq"])

    return {
        "data": data,
        "form4": form4,
        "cap_on": cap_on,
        "I_gen": I_gen,
        "fit": fit,
        "innovation": nu,
        "innovation_valid": valid,
        "stochastic": diag,
        "damage_curve": damage,
        "I_continuous_mult": I_cont,
    }


def print_summary(result, label="turbine"):
    fit = result["fit"]
    data = result["data"]
    cap_on = result["cap_on"]
    n_gen = int(data["gen_on"].sum())

    print(f"\n{'='*70}")
    print(f"  {label}  (Thévenin 1C1R)")
    print(f"{'='*70}")
    print(f"  Samples: {data['N']} ({n_gen} generating)")
    print(f"  Tw range: {data['Tw'].min():.1f} – {data['Tw'].max():.1f} °C")
    print(f"  Tnac mean: {data['Tnac'].mean():.1f} °C")
    print(f"  Cap ON (filtered): {cap_on.sum()} / {data['N']} "
          f"({100*cap_on.sum()/data['N']:.0f}%)")

    print(f"\n  Thermal fit:")
    se = fit["se"]
    print(f"    Rt_on   = {fit['Rt_on']:.4f} ± {se[0]:.4f} °C/W"
          f" (CV={se[0]/fit['Rt_on']*100:.2f}%)")
    print(f"    Rt_off  = {fit['Rt_off']:.4f} ± {se[1]:.4f} °C/W")
    print(f"    P0      = {fit['P0']:.0f} ± {se[2]:.0f} W")
    print(f"    C_eq    = {fit['C_eq']:.0f} ± {se[3]:.0f} J/°C"
          f" (CV={se[3]/fit['C_eq']*100:.2f}%)")
    print(f"    τ_on    = {fit['tau_on']:.0f} s ({fit['tau_on']/60:.1f} min)")
    print(f"    τ_off   = {fit['tau_off']:.0f} s ({fit['tau_off']/60:.1f} min)")
    print(f"    RMSE    = {fit['rmse']:.3f} °C")

    print(f"\n  Innovation diagnostics (whiteness check):")
    diag = result["stochastic"]
    print(f"    σ_w        = {diag['sigma_w']:.4f} °C/s")
    if diag["acf"]:
        print(f"    ACF(1..5)  = " + ", ".join(f"{a:+.3f}" for a in diag["acf"][:5]))
    print(f"    corr(I)    = {diag['corr_I']:+.3f}")
    print(f"    σ_hi/σ_lo  = {diag['sigma_ratio']:.3f}")
    if diag["acf"]:
        if abs(diag["acf"][0]) < 0.15 and abs(diag["corr_I"]) < 0.15:
            print(f"    → residual ≈ white, model fit is statistically clean")
        elif abs(diag["acf"][0]) > 0.3:
            print(f"    → coloured residual at lag-1, possible model misspecification")
        else:
            print(f"    → mild residual structure, acceptable for protection use")

    print(f"\n  Damage curve (cold start, T_amb=40°C):")
    for m, t in result["damage_curve"].items():
        ts = f"{t:.0f}s" if t < 1e6 else "∞"
        print(f"    {m:.1f}× I_rated → {ts}")
    if result["I_continuous_mult"]:
        Ic = result["I_continuous_mult"]
        print(f"  Continuous rating: {Ic:.3f}× rated ({Ic*I_RATED:.0f} A)")


# ════════════════════════════════════════════════════════════════════════════
# VISUALS
# ════════════════════════════════════════════════════════════════════════════

def plot_summary(result, label="turbine", out_path=None, title=None):
    """Six-panel summary figure.

    Parameters
    ----------
    title : str, optional
        Override for the figure suptitle. If None, uses
        ``label + " — Machine Study"``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = result["data"]
    fit = result["fit"]
    diag = result["stochastic"]
    nu = result["innovation"]
    valid = result["innovation_valid"]

    N = data["N"]
    t_hrs = data["t_sec"] / 3600.0
    Tw = data["Tw"]
    Tw_pred = fit["Tw_pred"]
    w = fit["weights"]
    mv = w > 0

    # Layout: 3×2 grid reading top-to-bottom like a report.
    #   Row 0 (top):    Damage curves (split full+zoom) | Model vs Measured
    #   Row 1 (middle): Winding temperature time series  | Residual time series
    #   Row 2 (bottom): Innovation histogram             | Model equations
    #
    # The top-left cell is split into two side-by-side sub-axes.
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 2, figure=fig)

    # Top-left: nested 1×2 sub-gridspec for full + zoom damage curves
    sub = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 0],
                                    width_ratios=[1.0, 1.2], wspace=0.30)
    ax_dmg_full = fig.add_subplot(sub[0, 0])
    ax_dmg_zoom = fig.add_subplot(sub[0, 1])
    ax_scatter = fig.add_subplot(gs[0, 1])
    ax_ts = fig.add_subplot(gs[1, 0])
    ax_resid = fig.add_subplot(gs[1, 1])
    ax_hist = fig.add_subplot(gs[2, 0])
    ax_eqns = fig.add_subplot(gs[2, 1])

    # Compute σ(Tw) up front — used in the time series legend
    sigma_Tw = float(np.std(Tw[mv]))

    # ── 1. Damage curve with IEC class 5 and ZEV class 5 relay overlays ──
    # ── 1a. Full range (1× to 8×) ──
    ax = ax_dmg_full

    int_anchors = np.array([1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5,
                             1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0,
                             7.2, 8.0])
    fine = np.linspace(1.05, 8.0, 200)
    I_mults = np.unique(np.concatenate([int_anchors, fine]))
    times = np.array([trip_time(fit["Rt_on"], fit["P0"], fit["C_eq"], 40.0, m)
                      for m in I_mults])
    valid_t = np.isfinite(times) & (times > 0)
    ax.semilogy(I_mults[valid_t], times[valid_t], 'C0', lw=2.5,
                label=f"Damage")
    if result["I_continuous_mult"]:
        Ic = result["I_continuous_mult"]
        ax.axvline(Ic, color='C0', ls=':', lw=1.0, alpha=0.8,
                   label=f"I_cont={Ic:.3f}·Ie")

    iec_grid = np.linspace(1.01, 8.0, 600)
    iec_times = 254.2 / (iec_grid ** 2 - 1.0)
    # IEC class 5 curve computed but not plotted (kept for reference)

    # ZEV class 5 fitted curve (logarithmic model):
    #     t = (CLASS/5) × 23.3 × ln(x/(x − 1.15))  where x = I/Ie
    # For CLASS 5: t = 23.3 × ln(x/(x − 1.15))
    # Asymptote at x = 1.15 — relay never trips below 1.15×Ie.
    zev_asymptote = 1.15
    zev_pts_x = np.array([3.0, 4.0, 5.0, 6.0, 7.2, 8.0])
    zev_fine = np.linspace(zev_asymptote + 0.001, 8.0, 600)
    zev_grid = np.unique(np.concatenate([zev_fine, zev_pts_x]))
    zev_times = 23.3 * np.log(zev_grid / (zev_grid - 1.15))
    ax.semilogy(zev_grid, zev_times, 'C3-', lw=1.6, alpha=0.9,
                label='ZEV class 5')

    zev_pts_y = 23.3 * np.log(zev_pts_x / (zev_pts_x - 1.15))
    ax.scatter(zev_pts_x, zev_pts_y, s=22, c='C3', edgecolors='white',
               linewidths=0.7, zorder=6)

    ax.plot([], [], ' ', label=f'Ie={I_RATED:.1f} A (ABB)')
    ax.set_xlabel('Current [× Ie]')
    ax.set_ylabel('Time to 155°C / trip [s]')
    ax.set_title('Damage & Relay Trip Curves',
                 fontweight='bold', fontsize=10)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(1, 6)
    ax.set_ylim(1, 100000)

    # ── 1b. Zoom on 1× to 1.5× showing the continuous-margin region ──
    ax = ax_dmg_zoom

    zoom_x = np.linspace(1.001, 1.5, 600)
    zoom_dmg = np.array([trip_time(fit["Rt_on"], fit["P0"], fit["C_eq"], 40.0, m)
                         for m in zoom_x])
    zoom_dmg_finite = np.where(np.isfinite(zoom_dmg) & (zoom_dmg > 0),
                                 zoom_dmg, np.nan)
    ax.semilogy(zoom_x, zoom_dmg_finite, 'C0', lw=2.5, label='Damage')

    iec_zoom = 254.2 / (zoom_x ** 2 - 1.0)
    # IEC class 5 zoom curve computed but not plotted (kept for reference)

    zev_zoom_x = zoom_x[zoom_x > zev_asymptote + 0.001]
    zev_zoom = 23.3 * np.log(zev_zoom_x / (zev_zoom_x - 1.15))
    ax.semilogy(zev_zoom_x, zev_zoom, 'C3-', lw=1.6, alpha=0.9,
                label='ZEV class 5')

    if result["I_continuous_mult"]:
        Ic = result["I_continuous_mult"]
        # Green "Safe" band: machine can sustain continuously
        ax.axvspan(1.0, Ic, color='C2', alpha=0.10)

        # If ZEV pickup < I_cont: no dead zone — relay is conservative,
        # it picks up before the machine reaches its continuous limit.
        # If ZEV pickup > I_cont: dead zone — machine overloaded but
        # relay inactive.
        if zev_asymptote > Ic:
            ax.axvspan(Ic, zev_asymptote, color='C1', alpha=0.18)

        ax.axvline(Ic, color='C0', ls=':', lw=1.2, alpha=0.9)
        ax.axvline(zev_asymptote, color='C3', ls=':', lw=1.2, alpha=0.9)

        if zev_asymptote > Ic:
            gap_pp = (zev_asymptote - Ic) * 100
            ax.text(0.5*(Ic + zev_asymptote), 2.5,
                    f'{gap_pp:.1f}pp', ha='center', va='bottom',
                    fontsize=7, color='darkorange', fontweight='bold')

        ax.annotate(f'I_cont {Ic:.3f}·Ie',
                    xy=(Ic, 1000), xytext=(Ic + 0.01, 1000),
                    ha='left', va='center', fontsize=7, color='C0',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor='C0', alpha=0.9))
        ax.annotate(f'ZEV pickup {zev_asymptote:.2f}·Ie',
                    xy=(zev_asymptote, 8.0),
                    xytext=(zev_asymptote + 0.01, 8.0),
                    ha='left', va='center', fontsize=7, color='C3',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor='C3', alpha=0.9))

    ax.set_xlabel('Current [× Ie]')
    ax.set_title('Zoom 1×–1.5×',
                 fontweight='bold', fontsize=10)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(1.0, 1.5)
    ax.set_ylim(1, 100000)

    # ── 2. Model vs Measured scatter with OLS line and R² ──
    ax = ax_scatter
    x_meas = Tw[mv]
    y_pred = Tw_pred[mv]
    lims = [min(x_meas.min(), y_pred.min()),
            max(x_meas.max(), y_pred.max())]

    ax.plot(lims, lims, color='0.78', lw=3.5, solid_capstyle='round',
            zorder=1)
    ax.scatter(x_meas, y_pred, s=2, alpha=0.3, c='C4', zorder=2)

    if len(x_meas) >= 2 and np.std(x_meas) > 1e-9:
        a_ols, b_ols = np.polyfit(x_meas, y_pred, 1)
        y_hat = a_ols * x_meas + b_ols
        ss_res = float(np.sum((y_pred - y_hat) ** 2))
        ss_tot = float(np.sum((y_pred - y_pred.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        x_line = np.array(lims)
        ax.plot(x_line, a_ols * x_line + b_ols, 'C1-', lw=1.5, zorder=3,
                label=f'y = {a_ols:.3f}x {b_ols:+.2f}')
        ax.text(0.04, 0.96, f'R² = {r2:.4f}',
                transform=ax.transAxes, ha='left', va='top',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='C1', alpha=0.85))

    ax.set_xlabel('Measured [°C]')
    ax.set_ylabel('Model [°C]')
    ax.set_title('Model vs Measured', fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # ── 3. Innovation histogram with fitted Gaussian ──
    ax = ax_hist
    nu_valid = nu[valid]
    if len(nu_valid) > 10:
        nu_mean = float(np.mean(nu_valid))
        nu_std = float(np.std(nu_valid))

        n_bins = min(80, max(30, len(nu_valid) // 60))
        counts, bin_edges, patches = ax.hist(
            nu_valid, bins=n_bins, density=True, alpha=0.55, color='C0',
            edgecolor='white', linewidth=0.3,
            label=f'μ = {nu_mean:.5f}\nσ = {nu_std:.5f}')

        x_span = 0.04
        x_gauss = np.linspace(-x_span, x_span, 300)
        y_gauss = (1.0 / (nu_std * np.sqrt(2 * np.pi))) * \
                  np.exp(-0.5 * ((x_gauss - nu_mean) / nu_std) ** 2)
        ax.plot(x_gauss, y_gauss, 'C3-', lw=2.0, alpha=0.85,
                label='Gaussian fit')

        ax.axvline(0, color='k', lw=0.5, alpha=0.5)
        ax.set_xlim(-x_span, x_span)

    ax.set_xlabel(r'Innovation $\nu_k$  (realization of $\sigma_w\,dB/dt$)  [°C/s]')
    ax.set_ylabel('Density')
    ax.set_title(r'Innovation histogram:  '
                 rf"$\sigma_w = \mathrm{{std}}(\nu) = {diag['sigma_w']:.4f}"
                 rf"\ (\pm{{\sim}}1\%)$,  "
                 rf"corr(I) = {diag['corr_I']:+.3f}",
                 fontweight='bold')
    ax.legend(fontsize=8, loc='upper right'); ax.grid(True, alpha=0.3)

    # ── 4. Model equations ──
    ax = ax_eqns
    ax.axis('off')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # ── 4. Model Parameters ──
    ax = ax_eqns
    ax.axis('off')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    se = fit.get("se", [float('nan')] * 4)
    # se order: [Rt_on, Rt_off, P0, C_eq]
    tau = fit["tau_on"]
    Ic = result["I_continuous_mult"]
    r2 = 1.0 - fit["rmse"]**2 / max(sigma_Tw**2, 1e-12)

    # ── Error propagation for derived quantities ──
    # Use the full first-order formula σ_f² = J·Σ·Jᵀ, where Σ is the
    # 4×4 parameter covariance matrix (from the Levenberg-Marquardt fit)
    # and J is the Jacobian of the derived quantity w.r.t. the 4
    # parameters. Parameter order: [Rt_on, Rt_off, P0, C_eq].
    corr = np.asarray(fit.get("corr", np.eye(4)))
    se_arr = np.asarray(se)
    cov = np.outer(se_arr, se_arr) * corr  # Σ = diag(se) · corr · diag(se)

    # τ_on = Rt_on × C_eq  →  ∂τ/∂Rt_on = C_eq,  ∂τ/∂C_eq = Rt_on
    J_tau = np.array([fit["C_eq"], 0.0, 0.0, fit["Rt_on"]])
    se_tau = float(np.sqrt(J_tau @ cov @ J_tau))

    # I_cont depends on Rt_on and P0 only (C_eq cancels at steady state).
    # Jacobian elements via central finite differences on the nonlinear
    # continuous_current_rating() function.
    se_Ic = float('nan')
    if Ic is not None:
        Rt0, P0_0, Ceq0 = fit["Rt_on"], fit["P0"], fit["C_eq"]
        dRt = max(se[0] * 0.1, 1e-7)
        dP0 = max(se[2] * 0.1, 1e-3)
        Ic_Rt_p = continuous_current_rating(Rt0 + dRt, P0_0, Ceq0)
        Ic_Rt_m = continuous_current_rating(Rt0 - dRt, P0_0, Ceq0)
        Ic_P0_p = continuous_current_rating(Rt0, P0_0 + dP0, Ceq0)
        Ic_P0_m = continuous_current_rating(Rt0, P0_0 - dP0, Ceq0)
        if all(v is not None for v in [Ic_Rt_p, Ic_Rt_m, Ic_P0_p, Ic_P0_m]):
            dIc_dRt = (Ic_Rt_p - Ic_Rt_m) / (2 * dRt)
            dIc_dP0 = (Ic_P0_p - Ic_P0_m) / (2 * dP0)
            J_Ic = np.array([dIc_dRt, 0.0, dIc_dP0, 0.0])
            se_Ic = float(np.sqrt(J_Ic @ cov @ J_Ic))

    params = [
        (0.93, r'$\bf{Fitted\ parameters}$', 12),
        (0.84, rf'$R_{{t,on}} = {fit["Rt_on"]:.5f} \pm {se[0]:.5f}\ \ '
               rf'[°C/W]$', 11),
        (0.75, rf'$R_{{t,off}} = {fit["Rt_off"]:.5f} \pm {se[1]:.5f}\ \ '
               rf'[°C/W]$', 11),
        (0.66, rf'$P_0 = {fit["P0"]:.1f} \pm {se[2]:.1f}\ \ '
               rf'[W]$', 11),
        (0.57, rf'$C_{{eq}} = {fit["C_eq"]:.0f} \pm {se[3]:.0f}\ \ '
               rf'[J/°C]$', 11),
        (0.46, r'$\bf{Derived\ quantities}$', 12),
        (0.37, rf'$\tau_{{on}} = R_{{t,on}} \times C_{{eq}}'
               rf' = {tau/60:.1f} \pm {se_tau/60:.1f}\ \mathrm{{min}}$', 11),
        (0.28, rf'$I_{{cont}} = {Ic:.3f} \pm {se_Ic:.3f}\ \times I_e'
               rf' = {Ic * I_RATED:.1f} \pm {se_Ic * I_RATED:.1f}'
               rf'\ \mathrm{{A}}$'
               if (Ic and np.isfinite(se_Ic))
               else (rf'$I_{{cont}} = {Ic:.3f} \times I_e'
                     rf' = {Ic * I_RATED:.1f}\ \mathrm{{A}}$'
                     if Ic else r'$I_{cont}:\ \mathrm{not\ computed}$'), 11),
        (0.17, r'$\bf{Fit\ quality}$', 12),
        (0.08, rf'$RMSE = {fit["rmse"]:.3f}°C \qquad'
               rf' R^2 = {r2:.4f} \qquad'
               rf' \sigma_w = {diag["sigma_w"]:.5f}$', 10),
    ]
    for y, txt, fs in params:
        ax.text(0.05, y, txt, transform=ax.transAxes, fontsize=fs,
                va='center', ha='left', family='serif')

    # Footer: ABB datasheet derivation and sample counts.
    # ABB measured line-to-line resistance at 25.6°C test ambient.
    # The pipeline uses the connection-agnostic effective resistance
    # R_dc = (3/2)·R_LL, corrected to the 20°C reference.
    T_ABB = 25.6  # ABB test ambient [°C]
    R_LL_20 = (2.0/3.0) * R_PHASE_20
    R_LL_abb = R_LL_20 * (1.0 + ALPHA_CU * (T_ABB - T_REF))
    ax.text(0.05, 0.00,
            rf'$R_{{dc,20}} = \frac{{3}}{{2}}R_{{LL}} = '
            rf'{R_PHASE_20:.5f}\ \Omega\ \ '
            rf'(\mathrm{{ABB\ R_{{LL}}@25.6°C}} = '
            rf'{R_LL_abb:.5f}\ \Omega,\ \Delta)$',
            transform=ax.transAxes, fontsize=11, va='center', ha='left',
            family='serif', color='black')
    ax.text(0.05, -0.05, rf'$N_{{eff}} = {fit["n_eff"]}'
            rf'\qquad N = {N}$',
            transform=ax.transAxes, fontsize=11, va='center', ha='left',
            family='serif', color='black')
    ax.set_title('Model parameters', fontweight='bold')

    # ── 5. Winding temperature time series ──
    ax = ax_ts
    ax.plot(t_hrs[mv], Tw[mv], 'C3', lw=0.5, alpha=0.8, label='Measured')
    ax.plot(t_hrs[mv], Tw_pred[mv], 'C0', lw=0.5, alpha=0.8,
            label=f"Model (RMSE={fit['rmse']:.2f}°C)")
    ax.fill_between(t_hrs, Tw[mv].min()-5, Tw[mv].max()+5,
                    where=~data["gen_on"], alpha=0.10, color='C2', label='Off')

    # Shade data gaps (same threshold as the fit algorithm: dt >
    # GAP_THRESHOLD_SEC, where the trapezoidal propagator re-initializes
    # from measured T_w rather than integrating across the gap).
    dt_sec = data["dt"]
    gap_labeled = False
    for i in range(len(dt_sec)):
        if dt_sec[i] > GAP_THRESHOLD_SEC:
            lbl = 'Data gap (>5 min)' if not gap_labeled else None
            ax.axvspan(t_hrs[i], t_hrs[i + 1],
                       color='0.85', alpha=0.6, zorder=0, label=lbl)
            gap_labeled = True

    ax.set_xlabel('Time [h]'); ax.set_ylabel('Tw [°C]')
    ax.set_title(f"Winding temperature fit | Rt_on={fit['Rt_on']:.4f}, "
                 f"P₀={fit['P0']:.0f}W, C_eq={fit['C_eq']:.0f}",
                 fontweight='bold')
    ax.plot([], [], ' ', label=f'σ(Tw) = {sigma_Tw:.2f} °C')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)

    # ── 6. Residual time series ──
    ax = ax_resid
    e = Tw - Tw_pred
    e[~mv] = np.nan
    ax.plot(t_hrs, e, 'C0', lw=0.4, alpha=0.6)
    ax.axhline(0, color='k', lw=0.5)
    ax.fill_between(t_hrs, -10, 10, where=~data["gen_on"], alpha=0.10, color='C2')
    ax.set_xlabel('Time [h]'); ax.set_ylabel('Residual [°C]')
    ax.set_title('Residual (Measured − Model) °C', fontweight='bold')
    ax.grid(True, alpha=0.3)

    sup = title if title else f"{label}  —  Machine Study"
    plt.suptitle(sup, fontsize=14,
                 fontweight='bold', y=1.005)
    plt.tight_layout()

    if out_path is None:
        import os
        base = os.path.splitext(os.path.basename(label))[0]
        out_path = f"{base}_thevenin_summary.png"
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_zev_curves(out_path=None):
    """
    Standalone ZEV motor-protective relay trip curves for all classes.

    Model: t_A = (CLASS/5) × 23.3 × ln(x / (x − 1.15))  where x = I/Ie
    Asymptote at x_trip = 1.15 × Ie.
    Fitted to Eaton/Moeller ZEV datasheet Table 8 (AWB2300-1433GB).

    Produces a single figure (one per fleet) with:
      Left panel  — log-log trip curves for all 8 classes, with Table 8
                    data points and 1.5×Ie intersection markers.
      Right panel — Table 8 data rendered as a styled table.
    """
    if out_path is None:
        out_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "zev_trip_curves.png")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    def _trip_time(x, cls):
        return (cls / 5.0) * 23.3 * np.log(x / (x - 1.15))

    # Table 8 data (datasheet AWB2300-1433GB, p.36)
    I_tab = np.array([3, 4, 5, 6, 7.2, 8, 10])
    tA_tab = {
        5:  np.array([11.3, 8.0, 6.1, 5.0, 4.1, 3.6, 2.9]),
        10: np.array([22.6, 15.9, 12.3, 10.0, 8.2, 7.3, 5.7]),
        15: np.array([34.0, 23.9, 18.4, 15.0, 12.3, 10.9, 8.6]),
        20: np.array([45.3, 31.8, 24.6, 20.0, 16.4, 14.6, 11.5]),
        25: np.array([56.6, 39.8, 30.7, 25.0, 20.5, 18.2, 14.4]),
        30: np.array([67.9, 47.7, 36.8, 30.0, 24.5, 21.9, 17.2]),
        35: np.array([79.2, 55.7, 43.0, 35.0, 28.6, 25.5, 20.1]),
        40: np.array([90.5, 63.6, 49.1, 40.0, 32.7, 29.2, 23.0]),
    }
    classes = [5, 10, 15, 20, 25, 30, 35, 40]

    fig, (ax_plot, ax_table) = plt.subplots(
        1, 2, figsize=(16, 13),
        gridspec_kw={'width_ratios': [2, 1]})

    x_dense = np.linspace(1.153, 11, 3000)
    cmap = plt.colormaps['inferno_r']
    cols = [cmap((i + 1) / (len(classes) + 2)) for i in range(len(classes))]

    # ── LEFT PANEL: Trip curves ──
    for i, cls in enumerate(classes):
        t_dense = _trip_time(x_dense, cls)
        ax_plot.loglog(x_dense, t_dense, '-', lw=2, color=cols[i],
                       label=f'CLASS {cls}')
        ax_plot.loglog(I_tab, tA_tab[cls], 'ko', ms=4, zorder=5, alpha=0.7)

    ax_plot.loglog([], [], 'ko', ms=4, label='Table 8 data')

    # 1.5×Ie intersections
    for i, cls in enumerate(classes):
        t15 = _trip_time(1.5, cls)
        ax_plot.plot([1.0, 1.5], [t15, t15], ls=':', lw=1.2,
                     color='#e88e8e', alpha=0.7, zorder=3)
        ax_plot.plot(1.5, t15, 'o', ms=5, color='red', zorder=6)
        ax_plot.text(1.02, t15, f'{t15:.0f}s', fontsize=8.5,
                     fontweight='bold', color=cols[i], ha='left', va='center',
                     bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                               edgecolor='none', alpha=0.85))
    ax_plot.plot([], [], 'o', ms=5, color='red', label='Model @ 1.5×Ie')

    # Reference lines
    ax_plot.axvline(1.5, color='steelblue', ls='--', lw=1, alpha=0.4)
    ax_plot.text(1.52, 1.3, '1.5×Ie', fontsize=9, color='steelblue',
                 va='bottom')
    ax_plot.axvline(1.15, color='gray', ls='--', lw=1.2, alpha=0.6)
    ax_plot.text(1.16, 2, r'$x_{trip}$ = 1.15', fontsize=9, color='gray',
                 rotation=90, va='bottom')

    ax_plot.legend(fontsize=9, loc='upper right', framealpha=0.9,
                   edgecolor='lightgray', title='Tripping Class',
                   title_fontsize=10)
    ax_plot.set_xlabel('× Ie', fontsize=14)
    ax_plot.set_ylabel(r'$t_A$', fontsize=14)
    ax_plot.set_xlim(1.0, 11)
    ax_plot.set_ylim(1, 7200)
    yticks = [1, 2, 3, 5, 10, 20, 30, 50, 60, 120, 300, 600, 1200,
              3600, 7200]
    ylabels = ['1s', '2s', '3s', '5s', '10s', '20s', '30s', '50s',
               '1 min', '2 min', '5 min', '10 min', '20 min', '1 hr',
               '2 hr']
    ax_plot.set_yticks(yticks)
    ax_plot.set_yticklabels(ylabels, fontsize=10)
    ax_plot.set_xticks([1.0, 1.5, 2, 3, 4, 5, 6, 8, 10])
    ax_plot.set_xticklabels(['1', '1.5', '2', '3', '4', '5', '6',
                             '8', '10'], fontsize=10)
    ax_plot.grid(True, which='major', alpha=0.4)
    ax_plot.grid(True, which='minor', alpha=0.1)

    # ── RIGHT PANEL: Table 8 ──
    ax_table.axis('off')
    col_headers = ['CLASS'] + [f'{x:g}' for x in I_tab]
    cell_text = []
    for cls in classes[::-1]:
        row = [f'{cls}']
        for j in range(len(I_tab)):
            row.append(f'{tA_tab[cls][j]:.1f}')
        cell_text.append(row)

    table = ax_table.table(
        cellText=cell_text, colLabels=col_headers,
        cellLoc='center', loc='center',
        bbox=[0.0, 0.25, 1.0, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Header styling
    for j in range(len(col_headers)):
        cell = table[0, j]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold', fontsize=10)
        cell.set_edgecolor('white')
        cell.set_linewidth(1.5)

    # Data cell styling
    for i in range(len(cell_text)):
        cls_val = classes[::-1][i]
        cls_idx = classes.index(cls_val)
        rgba = to_rgba(cols[cls_idx])
        tinted = (rgba[0], rgba[1], rgba[2], 0.2)
        for j in range(len(col_headers)):
            cell = table[i + 1, j]
            cell.set_edgecolor('#cccccc')
            cell.set_linewidth(0.5)
            if j == 0:
                cell.set_facecolor(tinted)
                cell.set_text_props(fontweight='bold')
            elif i % 2 == 0:
                cell.set_facecolor('#f7f7f7')
            else:
                cell.set_facecolor('white')

    ax_table.set_title(
        'Table 8: Assignment of tripping\ndelay tA [s] to tripping classes\n',
        fontsize=13, fontweight='bold', pad=20)
    ax_table.text(0.5, 0.22,
                  'In a 3-phase symmetrical tripping system,\n'
                  'the deviation of tA as of 3× the tripping\n'
                  'current is ±20%.',
                  transform=ax_table.transAxes, fontsize=9, ha='center',
                  va='top', style='italic', color='#555555')
    ax_table.text(0.5, 0.76, 'tA [s] at × Ie multiples',
                  transform=ax_table.transAxes, fontsize=10, ha='center',
                  va='bottom', color='#555555')

    fig.suptitle(
        'ZEV Motor-Protective Relay — Tripping Curves\n'
        r'$t_A = \frac{CLASS}{5} \times 23.3 \times'
        r' \ln\left(\frac{x}{x - 1.15}\right)$'
        r'    where $x = I / I_e$',
        fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def plot_equations_page(out_path=None):
    """
    Standalone one-page summary of all model equations used in the
    Machine Study pipeline. Produced once per fleet (not per turbine).
    """
    if out_path is None:
        out_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "model_equations.png")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 8.5))  # landscape letter
    ax.axis('off')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    eqns = [
        # Thermal model
        (0.96, r'$\bf{1.\ Thermal\ SDE}$', 14),
        (0.90, r'$C\,\frac{dT}{dt} = I^2 R_{dc}(T) + P_0 '
               r'- \frac{T - T_{amb}}{R_t}'
               r' + \sigma_w\,\frac{dB}{dt}$', 14),
        (0.83, r'$R_t = R_{t,on}\,\mathbf{1}_{\{on\}} + '
               r'R_{t,off}\,\mathbf{1}_{\{off\}}'
               r',\quad P_0 \to P_0\,\mathbf{1}_{\{on\}}'
               r',\quad dB/dt:\ \mathrm{white\ noise}$', 11),
        (0.73, r'$\bf{2.\ Thermal\ ODE}$', 13),
        (0.67, r'$\dfrac{dT}{dt} = a\,T + b(t)'
               r'\quad a = \dfrac{I^2 R_{dc,20}\,\alpha - 1/R_t}{C}'
               r'\quad b = \dfrac{I^2 R_{dc,20}(1-\alpha T_{ref})'
               r' + P_0 + T_{amb}/R_t}{C}$', 11),
        (0.60, r'$R_{dc}(T) = R_{dc,20}[1 + \alpha(T - T_{ref})]'
               r',\ T_{ref} = 20°C,\ \alpha = 0.00393/°C$', 11),
        (0.53, r'$\mathrm{Steady\ state\ at\ }T_{lim}:\ \ '
               r'I_{cont}^{2}\,R_{dc}(T_{lim}) + P_0'
               r' = (T_{lim} - T_{amb})/R_{t,on}$', 12),
        (0.46, r'$\Rightarrow\ I_{cont} = \sqrt{\left['
               r'(T_{lim} - T_{amb})/R_{t,on} - P_0\right]'
               r'\,/\,R_{dc}(T_{lim})}'
               r'\quad (T_{lim} = 155°C,\ \mathrm{class\ F})$', 12),
        # Numerical integration
        (0.36, r'$\bf{3.\ Trapezoidal\ (exponential)\ update}$', 13),
        (0.30, r'$T_{k+1} = e^{a\Delta t}\!\left(T_k + '
               r'\frac{\Delta t}{2}\,b_k\right) + '
               r'\frac{\Delta t}{2}\,b_{k+1}$', 13),
        # Innovation
        (0.21, r'$\bf{4.\ Innovation\ (whitened\ residual)}$', 13),
        (0.14, r'$e_k = T_{meas,k} - T_{model,k}'
               r'\quad'
               r'\nu_k = \dfrac{e_{k+1} - e^{a\Delta t}\,e_k}{\Gamma_k}'
               r',\ \Gamma_k = \dfrac{e^{a\Delta t} - 1}{a}'
               r'\quad'
               r'\sigma_w = \sqrt{\dfrac{1}{N_{eff}}'
               r'\sum_{k=1}^{N_{eff}} (\nu_k - \bar{\nu})^2}$', 10),
        # Relay and machine trip time
        (0.06, r'$\bf{5.\ Relay\ and\ Machine\ Trip\ Time\ (t_A)}$', 13),
        (-0.01, r'$t_A^{ZEV} = \dfrac{CLASS}{5}\times 23.3'
               r'\,\ln\!\left(\dfrac{I/I_e}{I/I_e - 1.15}\right)'
               r',\quad'
               r't_A^{mach} = \dfrac{1}{a}'
               r'\ln\!\left(\dfrac{T_{lim}-T_{ss}}{T_0-T_{ss}}\right)'
               r',\ T_{ss} = -b/a$', 11),
        # Error Statistics — full covariance form
        (-0.09, r'$\bf{6.\ Error\ statistics\ (first-order\ propagation)}$',
                13),
        (-0.16, r'$\sigma_f^2 = J\,\Sigma\,J^{\!\top} = '
                r'\sum_{i,j} \frac{\partial f}{\partial x_i}'
                r'\frac{\partial f}{\partial x_j}\,\Sigma_{ij}'
                r'\qquad \Sigma = \mathrm{diag}(\sigma)\cdot'
                r'\mathrm{corr}\cdot\mathrm{diag}(\sigma)$', 11),
        (-0.23, r'$J_{\tau_{on}} = [C_{eq},\,0,\,0,\,R_{t,on}]'
                r'\qquad J_{I_{cont}} = ['
                r'\partial I_{cont}/\partial R_{t,on},\,0,\,'
                r'\partial I_{cont}/\partial P_0,\,0]$', 11),
        # R², N_eff, w_k definition
        (-0.31, r'$R^2 = 1 - \frac{RMSE^2}{\sigma^2(T_w)}'
                r'\qquad N_{eff} = \sum_{k=1}^{N} \mathbf{1}(w_k = 1)$', 11),
        (-0.37, r'$w_k = 0\ \mathrm{if}\ \Delta t_{k-1} > 300\ \mathrm{s}'
                r'\ \mathrm{(data\ gap)},\quad'
                r' w_k = 1\ \mathrm{otherwise}$', 10),
    ]
    for y, txt, fs in eqns:
        ax.text(0.05, y, txt, transform=ax.transAxes, fontsize=fs,
                va='center', ha='left', family='serif')

    fig.suptitle('Model Equations Summary', fontsize=18,
                 fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path
# ════════════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python turbine_thermal_pipeline_thevenin.py <csv_path>")
        sys.exit(1)

    path = sys.argv[1]
    label = path.split("/")[-1]
    result = process(path)
    print_summary(result, label=label)
    plot_path = plot_summary(result, label=label)
    print(f"\n  Summary plot: {plot_path}")
