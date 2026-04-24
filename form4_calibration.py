"""
Form 4 fleet calibration.

Reproduces the fleet-wide Form 4 parameters used by the Thévenin
thermal pipelines. Takes a list of SCADA CSVs as input and produces four
numbers as output:

    Q0_FORM4   - no-load magnetising reactive         [kVAR]
    K_FORM4    - leakage coefficient                  [kVAR/kW^2]
    Q_CAP_NOM  - cap bank reactive at V_NOM           [kVAR]
    V_NOM      - nominal phase voltage                [V]

Methodology (from the prior fleet analysis):
  1. Load each turbine CSV, extract (P, Q, V, PF_meter_raw).
  2. Detect cap state per sample from |PF_meter_raw| > PF_THRESHOLD,
     then apply a refractory filter (min run length = MIN_RUN) to
     suppress single-sample spurious flips.
  3. Aggregate across the fleet the samples where the turbine is
     generating (P > P_MIN_GEN) and V > V_MIN_VALID.
  4. Compute V_nom as the mean phase voltage across those samples.
  5. ABB-optimal fit:
       Sweep a trial cap-bank reactive Q_cap. For each trial value,
       decompensate the measured Q by subtracting cap_on * Q_cap, then
       fit Q0, k by linear regression of  -Q_decomp = Q0 + k * P^2.
       Score the resulting Form 4 curve by RMS PF error against the
       ABB datasheet points. The (Q_cap, Q0, k) that minimises ABB
       error is the fleet calibration.
"""

import sys
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


# ----- configuration -----
PF_THRESHOLD = 9000       # |PF x 10000| above this -> cap ON
MIN_RUN = 3               # refractory filter minimum run length
P_MIN_GEN = 1.0           # kW, minimum generating power for inclusion
V_MIN_VALID = 100.0       # V,  minimum valid phase voltage
I_RELIABLE = 20.0         # A,  below this, PF meter is unreliable (CT noise floor)

# Binned-fit configuration (option 1: fit on power-bin medians)
BIN_WIDTH_KW = 2.0        # bin width in kW
BIN_P_MIN = 1.0           # lowest bin edge [kW]
BIN_P_MAX = 100.0         # highest bin edge [kW]
BIN_MIN_SAMPLES = 20      # minimum samples per bin to include in the fit

# ABB datasheet PF points (M3AA 250SMA 4G 400V 50Hz 55kW)
ABB_P = np.array([1.04, 15.30, 29.52, 44.04, 58.25])    # kW
ABB_PF = np.array([0.04, 0.51, 0.72, 0.81, 0.84])

# Turbines whose cap-OFF samples are dropped from the fleet fit by default.
#
# Background: T3 (10_0_153_10.csv) has cap-OFF samples that sit on a parallel
# curve 5-9 kVAR below the fleet Form 4 curve, with the offset growing with
# power. This is NOT a classifier error -- the voting + EM converged stably
# and T3's cap-ON samples sit cleanly on the main cloud with the rest of the
# fleet. The cap-OFF population signature is consistent with residual/partial
# compensation in the nominally-OFF state (a second always-on cap bank, a
# partial contactor disconnect, a stuck contactor, or similar installation
# anomaly that we cannot diagnose from SCADA alone).
#
# Why this mechanism exists at all: when we discovered T3's behavior, we
# wanted a clean way to characterise its impact on the fleet calibration
# without per-case patches scattered through the code. Excluding the
# affected turbine's cap-OFF samples (while keeping its cap-ON samples,
# which are well-behaved) was the minimum intervention that produced a
# defensible fit without throwing T3 out entirely.
#
# Why it is DISABLED by default: subsequent analysis showed that
#   (1) the binned-median fit is already robust to T3's outlier samples
#       -- they only contribute as part of the median in each power bin,
#       not as individual leveraged points;
#   (2) the downstream Thevenin thermal RMSE on every turbine is
#       indistinguishable between the included and excluded cases
#       (delta < 0.01 deg C, well inside numerical noise -- the gauge
#       freedom in (Rt_on, P0, C_eq) absorbs the small Form 4 shift);
#   (3) per-case data exclusions are hard to defend in production and
#       risk hiding genuine fleet heterogeneity behind a knob.
# So we leave the mechanism in place but inactive: the calibration uses
# all samples from all turbines by default, and the residual T3 cap-OFF
# anomaly remains visible in the diagnostic plot rather than hidden.
#
# When to re-enable: add "10_0_153_10.csv" (or any other turbine
# basename) to the list below if you find a specific study where the
# Form 4 calibration error needs to drop into the type-test range
# (e.g. comparing fitted Q0/k against the ABB datasheet for warranty
# purposes), or if a future turbine exhibits the same residual-comp
# pathology and you want to exclude it without modifying calibrate_form4
# call sites.
DEFAULT_EXCLUDE_CAP_OFF = [
    # "10_0_153_10.csv",   # T3 cap-OFF -- see comment above
]


# ----- per-sample helpers (copied verbatim from the thermal pipeline) -----

def refractory_filter(raw_state, min_run=MIN_RUN):
    """Suppress brief state changes; accept only runs of >= min_run."""
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
    cap_raw = np.abs(PF_meter_raw) > PF_THRESHOLD
    return refractory_filter(cap_raw, min_run=MIN_RUN)


# ----- data loader -----

def load_turbine(path):
    """Return a dict with the columns needed for Form 4 calibration."""
    df = pd.read_csv(path)
    df['time'] = pd.to_datetime(df['timestamp'], format='mixed')
    df = df.sort_values('time').reset_index(drop=True)

    P = df['@GV.HRR_kW'].values / 10.0
    Q = df['@GV.HRR_kVAR'].values / 10.0
    V = (df['@GV.primaryLovatoReadings.L1PhaseVoltage'].values
         + df['@GV.primaryLovatoReadings.L2PhaseVoltage'].values
         + df['@GV.primaryLovatoReadings.L3PhaseVoltage'].values) / 3.0 / 100.0
    I_avg = (df['@GV.primaryLovatoReadings.L1Current'].values
             + df['@GV.primaryLovatoReadings.L2Current'].values
             + df['@GV.primaryLovatoReadings.L3Current'].values) / 3.0 / 10.0
    PF_meter_raw = df['@GV.primaryLovatoReadings.EqvPowerFactor'].values

    cap_on = detect_cap_state(PF_meter_raw)

    # Generation mask: matches turbine_thermal_pipeline.load_scada convention
    # (off only if P=Q=0 for two consecutive samples)
    N = len(P)
    pq0 = (np.abs(P) < 0.5) & (np.abs(Q) < 0.5)
    gen_on = np.ones(N, dtype=bool)
    for i in range(1, N):
        gen_on[i] = not (pq0[i] and pq0[i - 1])

    return {'P': P, 'Q': Q, 'V': V, 'I_avg': I_avg,
            'PF_meter_raw': PF_meter_raw,
            'cap_on': cap_on, 'gen_on': gen_on}


# ----- fitting -----

def fit_Q0_k_given_Qcap(P_m, Q_meas, cap_on, Q_cap,
                        binned=False, bin_edges=None):
    """
    For a given Q_cap, decompensate Q and fit (Q0, k) by linear regression
    of  -Q_decomp = Q0 + k * P^2.

    Two modes:

      binned=False  ->  raw per-sample regression (every generating sample
                        contributes equally; fit is dominated by whichever
                        operating regime the fleet visits most often).

      binned=True   ->  fit on the median decompensated Q within each
                        power bin. All bins with >= BIN_MIN_SAMPLES samples
                        contribute one data point to the regression. This
                        weighs every operating regime equally regardless
                        of how often the fleet visits it, and is robust
                        to outlier turbines that dominate one slice of
                        the operating range.

    The reported RMSE is always computed against the full sample set so
    the two modes are directly comparable.
    """
    Q_decomp = Q_meas - cap_on * Q_cap
    y_full = -Q_decomp
    x_full = P_m ** 2

    if binned:
        if bin_edges is None:
            bin_edges = np.arange(BIN_P_MIN, BIN_P_MAX + BIN_WIDTH_KW,
                                  BIN_WIDTH_KW)
        # Per-bin median of -Q_decomp and per-bin sample count
        bin_idx = np.digitize(P_m, bin_edges) - 1
        n_bins = len(bin_edges) - 1
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        x_bins = []
        y_bins = []
        for b in range(n_bins):
            sel = bin_idx == b
            if int(sel.sum()) >= BIN_MIN_SAMPLES:
                x_bins.append(centers[b] ** 2)
                y_bins.append(float(np.median(y_full[sel])))
        if len(x_bins) < 2:
            # Fall back to raw fit if we somehow have fewer than 2 bins
            x = x_full
            y = y_full
        else:
            x = np.array(x_bins)
            y = np.array(y_bins)
    else:
        x = x_full
        y = y_full

    n = len(y)
    sx = x.sum()
    sy = y.sum()
    sxx = (x * x).sum()
    sxy = (x * y).sum()
    k = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    Q0 = (sy - k * sx) / n

    # Always report RMSE against the full sample set so the two modes
    # are directly comparable.
    resid_full = y_full - (Q0 + k * x_full)
    rmse = float(np.sqrt(np.mean(resid_full ** 2)))
    return Q0, k, rmse


def abb_pf_error(Q0, k):
    """RMS error of Form 4 PF against the 5 ABB datasheet points."""
    pf_pred = ABB_P / np.sqrt(ABB_P ** 2 + (Q0 + k * ABB_P ** 2) ** 2)
    return float(np.sqrt(np.mean((pf_pred - ABB_PF) ** 2)))


def _reclassify_turbine(turb, Q0, k, Q_cap):
    """
    Confidence-weighted voting reclassification of one turbine's samples.

    Two classifiers vote on each generating sample:

      Classifier A — PF threshold:
          vote_on  = |PF_meter| > 0.9
          conf     = |(|PF_meter| - 0.9)| / 0.5    (clipped to [0, 1])
          Strong when the metered PF is far from the 0.9 boundary,
          weak when it sits near the threshold.
          Attenuated at low current: conf *= clip(I_avg / I_RELIABLE, 0, 1).
          Below I_RELIABLE (20 A) the Lovato PF reading is unreliable
          (near the CT noise floor) and should not override the EM
          classifier's reactive-power evidence.

      Classifier B — Form 4 distance (EM):
          vote_on  = closer to cap-ON branch than to cap-OFF branch
          conf     = ||d_off - d_on|| / Q_cap      (clipped to [0, 1])
          Strong when the sample sits clearly on one branch,
          weak when it sits midway between the two.

    Decision rule:
      - If A and B agree, accept the agreed label.
      - If they disagree, use whichever vote has higher confidence
        (the less-confident classifier is "penalised" by being overruled).
      - Run the refractory filter on the post-vote sequence to suppress
        single-sample flickers.
      - Non-generating samples are NOT reclassified: their cap_on value
        is held at its initial (load-time) value. Those samples are
        excluded from the Form 4 fit anyway.
    """
    P = turb['P']
    Q = turb['Q']
    P_sq = P ** 2

    # ---- Classifier B: Form 4 distance (EM) ----
    Q_off_pred = -(Q0 + k * P_sq)
    Q_on_pred = Q_off_pred + Q_cap
    d_off = np.abs(Q - Q_off_pred)
    d_on = np.abs(Q - Q_on_pred)
    em_vote_on = d_on < d_off
    em_conf = np.clip(np.abs(d_off - d_on) / max(Q_cap, 1e-6), 0.0, 1.0)

    # ---- Classifier A: PF threshold (current-weighted) ----
    pf_abs = np.abs(turb['PF_meter_raw']) / 10000.0
    pf_vote_on = pf_abs > 0.9
    pf_conf = np.clip(np.abs(pf_abs - 0.9) / 0.5, 0.0, 1.0)
    # Attenuate PF confidence at low current where the meter is unreliable
    current_weight = np.clip(turb['I_avg'] / I_RELIABLE, 0.0, 1.0)
    pf_conf = pf_conf * current_weight

    # ---- Vote ----
    agree = (pf_vote_on == em_vote_on)
    em_wins = em_conf > pf_conf
    voted = np.where(agree, em_vote_on,
                     np.where(em_wins, em_vote_on, pf_vote_on))

    # Refractory filter on the full sequence to suppress flicker
    voted = refractory_filter(voted, min_run=MIN_RUN)

    # ---- Gate by gen_on ----
    # Non-generating samples retain their original cap_on value.
    result = turb['cap_on'].copy()
    gen = turb['gen_on']
    result[gen] = voted[gen]
    return result


def _aggregate(turbines, exclude_cap_off=None):
    """
    Concatenate per-turbine arrays into fleet-wide vectors.

    `exclude_cap_off` is an optional iterable of turbine names (or labels)
    whose cap-OFF samples should be dropped from the aggregate. Their
    cap-ON samples are still included. This is the mechanism for handling
    turbines with unreliable "cap OFF" state (e.g. T3, whose cap-OFF
    samples sit on a parallel curve that no single Q_cap can explain,
    suggesting residual compensation in the nominally-OFF state).

    Returns
    -------
    P, Q, V, cap, gen, label
        label is an object array of per-sample turbine labels, parallel
        to P/Q/V, useful for per-turbine colouring in diagnostic plots.
    """
    exclude_set = set(exclude_cap_off or [])
    P_parts, Q_parts, V_parts, cap_parts, gen_parts, lbl_parts = \
        [], [], [], [], [], []
    for t in turbines:
        name = t.get('name') or t.get('label') or ''
        # Prefer short display label if present, fall back to basename
        display = t.get('label') or name
        if name in exclude_set:
            # Keep only cap-ON samples from this turbine
            keep = t['cap_on']
        else:
            keep = np.ones(len(t['P']), dtype=bool)
        P_parts.append(t['P'][keep])
        Q_parts.append(t['Q'][keep])
        V_parts.append(t['V'][keep])
        cap_parts.append(t['cap_on'][keep])
        gen_parts.append(t['gen_on'][keep])
        lbl_parts.append(np.full(int(keep.sum()), display, dtype=object))
    P = np.concatenate(P_parts)
    Q = np.concatenate(Q_parts)
    V = np.concatenate(V_parts)
    cap = np.concatenate(cap_parts).astype(float)
    gen = np.concatenate(gen_parts)
    label = np.concatenate(lbl_parts)
    return P, Q, V, cap, gen, label


def _fit_once(turbines, binned=False, exclude_cap_off=None):
    """
    Given per-turbine arrays with whatever cap_on classification is
    currently attached, aggregate them, apply the masks, and return the
    ABB-optimal (Q0, k, Q_cap) via the Q_cap sweep + linear regression.
    """
    P_all, Q_all, V_all, cap_all, gen_all, lbl_all = _aggregate(
        turbines, exclude_cap_off=exclude_cap_off)
    mask = gen_all & (P_all > P_MIN_GEN)
    P_m = np.abs(P_all[mask])
    Q_m = Q_all[mask]
    V_m = V_all[mask]
    cap_m = cap_all[mask]
    lbl_m = lbl_all[mask]
    V_nom = float(V_m.mean())

    bin_edges = np.arange(BIN_P_MIN, BIN_P_MAX + BIN_WIDTH_KW, BIN_WIDTH_KW)

    def abb_cost(Q_cap):
        Q0, k, _ = fit_Q0_k_given_Qcap(P_m, Q_m, cap_m, Q_cap,
                                       binned=binned, bin_edges=bin_edges)
        return abb_pf_error(Q0, k)

    Q_cap_grid = np.linspace(15.0, 30.0, 301)
    costs = np.array([abb_cost(q) for q in Q_cap_grid])
    i0 = int(np.argmin(costs))
    lo = Q_cap_grid[max(0, i0 - 2)]
    hi = Q_cap_grid[min(len(Q_cap_grid) - 1, i0 + 2)]
    res = minimize_scalar(abb_cost, bounds=(lo, hi), method='bounded',
                          options={'xatol': 1e-6})
    Q_cap_opt = float(res.x)
    Q0_opt, k_opt, rmse = fit_Q0_k_given_Qcap(
        P_m, Q_m, cap_m, Q_cap_opt, binned=binned, bin_edges=bin_edges)
    return {
        'Q0': Q0_opt, 'k': k_opt, 'Q_cap': Q_cap_opt, 'V_nom': V_nom,
        'rmse_kvar': rmse,
        'abb_err': abb_pf_error(Q0_opt, k_opt),
        'mask': mask, 'P_m': P_m, 'Q_m': Q_m, 'V_m': V_m, 'cap_m': cap_m,
        'lbl_m': lbl_m,
    }


_USE_DEFAULT = object()  # sentinel so callers can pass None to disable


def calibrate_form4(csv_paths, em_iter=8, em_tol=1e-5, binned=True,
                    exclude_cap_off=_USE_DEFAULT):
    """
    Fleet Form 4 calibration with EM-style reclassification refinement.

    `exclude_cap_off` controls which turbines' cap-OFF samples are
    dropped from the fit:
      - Omitted / _USE_DEFAULT: use DEFAULT_EXCLUDE_CAP_OFF from the top
        of the file (currently T3 = 10_0_153_10.csv).
      - None or []: include every sample from every turbine.
      - [name1, name2, ...]: drop cap-OFF samples from the named turbines.

    See the DEFAULT_EXCLUDE_CAP_OFF comment in this file for the physical
    justification for the default list.
    """
    if exclude_cap_off is _USE_DEFAULT:
        exclude_cap_off = DEFAULT_EXCLUDE_CAP_OFF
    print(f"Loading {len(csv_paths)} turbines...")
    turbines = []
    for i, p in enumerate(csv_paths):
        d = load_turbine(p)
        d['name'] = p.split('/')[-1]
        d['label'] = f"M{i + 1}"
        turbines.append(d)
        print(f"  {d['label']} ({d['name']}): {len(d['P'])} samples, "
              f"{int(d['cap_on'].sum())} cap-ON (PF threshold)")

    mode_str = "binned-median" if binned else "per-sample"
    excl_str = (f", excluding cap-OFF from: {list(exclude_cap_off)}"
                if exclude_cap_off else "")
    print(f"\n  Fit mode: {mode_str}{excl_str}")

    # ---- initial fit ----
    fit = _fit_once(turbines, binned=binned, exclude_cap_off=exclude_cap_off)
    print(f"\n  Initial fit (PF-threshold labels):")
    print(f"    Q0={fit['Q0']:.4f}  k={fit['k']:.6f}  "
          f"Q_cap={fit['Q_cap']:.3f}  ABB err={fit['abb_err']:.4f}")

    # ---- EM loop ----
    history = [dict(Q0=fit['Q0'], k=fit['k'], Q_cap=fit['Q_cap'],
                    abb_err=fit['abb_err'], reclassified=0,
                    n_cap_on=int(fit['cap_m'].sum()))]

    for iteration in range(1, em_iter + 1):
        # E-step: reclassify each turbine using current params
        total_changed = 0
        for t in turbines:
            old = t['cap_on'].copy()
            t['cap_on'] = _reclassify_turbine(
                t, fit['Q0'], fit['k'], fit['Q_cap'])
            total_changed += int(np.sum(old != t['cap_on']))

        # M-step: refit
        new_fit = _fit_once(turbines, binned=binned,
                            exclude_cap_off=exclude_cap_off)

        dQ0 = abs(new_fit['Q0'] - fit['Q0'])
        dk = abs(new_fit['k'] - fit['k'])
        dQcap = abs(new_fit['Q_cap'] - fit['Q_cap'])
        fit = new_fit

        history.append(dict(
            Q0=fit['Q0'], k=fit['k'], Q_cap=fit['Q_cap'],
            abb_err=fit['abb_err'], reclassified=total_changed,
            n_cap_on=int(fit['cap_m'].sum())))

        print(f"  EM iter {iteration}: {total_changed:>5} reclassified, "
              f"Q0={fit['Q0']:.4f}  k={fit['k']:.6f}  "
              f"Q_cap={fit['Q_cap']:.3f}  ABB err={fit['abb_err']:.4f}")

        if total_changed == 0 or (dQ0 < em_tol and dk < em_tol
                                   and dQcap < em_tol):
            print(f"  Converged at iteration {iteration}")
            break

    return {
        'Q0': fit['Q0'],
        'k': fit['k'],
        'Q_cap': fit['Q_cap'],
        'V_nom': fit['V_nom'],
        'rmse_kvar': fit['rmse_kvar'],
        'abb_err': fit['abb_err'],
        'n_samples': int(fit['mask'].sum()),
        'n_cap_on': int(fit['cap_m'].sum()),
        'em_history': history,
        'binned': binned,
        # Raw aggregated fleet samples (for plotting / downstream validation)
        '_P': fit['P_m'],
        '_Q': fit['Q_m'],
        '_V': fit['V_m'],
        '_cap_on': fit['cap_m'],
        '_lbl': fit['lbl_m'],
    }


def print_result(r):
    print("\n" + "=" * 60)
    print("  FORM 4 FLEET CALIBRATION RESULT")
    print("=" * 60)
    print(f"  Q0_FORM4  = {r['Q0']:.4f}   kVAR")
    print(f"  K_FORM4   = {r['k']:.6f} kVAR/kW^2")
    print(f"  Q_CAP_NOM = {r['Q_cap']:.3f}  kVAR")
    print(f"  V_NOM     = {r['V_nom']:.2f}   V")
    print(f"\n  Aggregate RMSE (decompensated Q): {r['rmse_kvar']:.3f} kVAR")
    print(f"  ABB PF RMS error               : {r['abb_err']:.4f}")
    print(f"  Samples used                   : {r['n_samples']} "
          f"({r['n_cap_on']} cap-ON)")


def plot_result(r, out_path=None):
    """
    Two-panel summary plot:
      (left)  Q vs P scatter coloured by cap state, with the Form 4
              curves for cap-OFF and cap-ON overlaid.
      (right) Machine PF vs P: measured samples (after decompensating
              cap-ON Q by the fitted Q_cap), the fitted Form 4 curve,
              the previously published curve for reference, and the
              5 ABB datasheet points as ground truth.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    P = r['_P']; Q = r['_Q']; V = r['_V']; cap = r['_cap_on']
    lbl = r.get('_lbl', None)
    Q0 = r['Q0']; k = r['k']; Q_cap = r['Q_cap']; V_nom = r['V_nom']

    # Decompensate Q for cap-ON samples -> generator Q (Q_cap offset removed)
    Q_decomp = Q - cap * Q_cap
    PF_meas = np.abs(P) / np.sqrt(P ** 2 + Q_decomp ** 2 + 1e-12)

    # ── Outlier note ────────────────────────────────────────────────
    # Dead samples (P=0, Q=0, I=0, PF≈0) are excluded upstream by
    # the broadened pq0 filter in load_turbine() / load_scada(),
    # which marks them as gen_on=False. No additional plot-level
    # filtering is needed.
    n_removed = 0

    # With Q_cap offset removed, both cap states collapse to a single curve
    P_curve = np.linspace(1.0, 100.0, 400)
    Q_curve = -(Q0 + k * P_curve ** 2)

    # Right-panel reference curves
    PF_fit = P_curve / np.sqrt(P_curve ** 2 + (Q0 + k * P_curve ** 2) ** 2)

    on = cap > 0.5
    off = ~on

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))

    # ----- LEFT: decompensated Q vs P, coloured per turbine -----
    if lbl is not None:
        unique_turbs = sorted(set(lbl.tolist()),
                              key=lambda x: int(x[1:]))
        cmap = plt.get_cmap('tab10')
        for i, tname in enumerate(unique_turbs):
            m = (lbl == tname)
            n_pts = int(m.sum())
            axL.scatter(np.abs(P[m]), Q_decomp[m], s=2, alpha=0.35,
                        c=[cmap(i % 10)],
                        label=f'{tname} (n={n_pts})')
    else:
        axL.scatter(np.abs(P), Q_decomp, s=2, alpha=0.25, c='C0',
                    label=f'Fleet (n={len(P)})')

    axL.plot(P_curve, Q_curve, 'k-', lw=2.5,
             label=f'Form 4: −(Q\u2080 + k·P²)  '
                   f'(Q\u2080={Q0:.2f}, k={k:.5f})')
    axL.set_xlabel('P [kW]')
    axL.set_ylabel('Q_decomp = Q − cap·Q_cap  [kVAR]')
    axL.set_title(f'Machine Reactance Scatter Q_cap {Q_cap:.2f} kVAR offset removed',
                  fontweight='bold')
    axL.legend(fontsize=8, loc='lower left', ncol=2)
    axL.grid(True, alpha=0.3)
    axL.set_xlim(0, 100)

    # ─── ABB Type Test validation ──────────────────────────────────
    # Q₀ from no-load test: Q_NL = √(S² − P²)
    V_NL_ABB, I_NL_ABB, P_NL_ABB = 400.8, 35.6, 1040.0
    S_NL_ABB = np.sqrt(3) * V_NL_ABB * I_NL_ABB
    Q_NL_ABB = np.sqrt(S_NL_ABB**2 - P_NL_ABB**2) / 1000.0

    # k from locked-rotor test via equivalent circuit extraction
    # X_eq = 1.2369 Ω (X1+X2', per phase delta, from LR test)
    # I2'_rated = 48.62 A (from phasor decomposition at rated)
    X_eq_LR = 1.2369
    I2_prime_rated = 48.62
    P_rated_kW = 58.25
    k_ABB = 3 * X_eq_LR * I2_prime_rated**2 / (P_rated_kW**2 * 1000)

    # ABB-derived Form 4 curve on the scatter
    Q_curve_ABB = -(Q_NL_ABB + k_ABB * P_curve**2)
    axL.plot(P_curve, Q_curve_ABB, '--', color='#CC3311', lw=2.0,
             label=f'ABB-derived: Q\u2080={Q_NL_ABB:.2f}, k={k_ABB:.5f}',
             zorder=7)

    # Marker for SCADA fit intercept (at P = 0)
    axL.plot(0, -Q0, 'o', markersize=12,
             markerfacecolor='#1F4E79', markeredgecolor='white',
             markeredgewidth=1.5, zorder=8, clip_on=False,
             label=f'SCADA Q\u2080 = {Q0:.2f} kVAR')
    # Marker for ABB no-load measurement (nested inside blue circle)
    axL.plot(0, -Q_NL_ABB, 's', markersize=6,
             markerfacecolor='#CC3311', markeredgecolor='white',
             markeredgewidth=0.8, zorder=9, clip_on=False,
             label=f'ABB no-load calculated Q\u2080 = {Q_NL_ABB:.2f} kVAR')

    # Re-add legend to include the ABB curve
    axL.legend(fontsize=7, loc='lower left', ncol=2)

    # PF comparison at the 5 ABB test points
    ABB_TEST = [(1.04, 0.04), (15.30, 0.51), (29.52, 0.72),
                (44.04, 0.81), (58.25, 0.84)]
    def _pf(P, q0, kk):
        Q = q0 + kk * P**2
        return P / np.sqrt(P**2 + Q**2)

    pf_abb_errs = [_pf(p, Q_NL_ABB, k_ABB) - pf_ds for p, pf_ds in ABB_TEST]
    pf_f4_errs = [_pf(p, Q0, k) - pf_ds for p, pf_ds in ABB_TEST]
    rms_abb = float(np.sqrt(np.mean(np.array(pf_abb_errs)**2)))
    rms_f4 = float(np.sqrt(np.mean(np.array(pf_f4_errs)**2)))

    # Build compact validation table with aligned columns
    tbl = (
        r'$\bf{ABB\ Type\ Test\ validation}$' + '\n'
        '──────────────────────────────────\n'
        '           ABB      SCADA\n'
        '           test       fit\n'
        f' Q\u2080     {Q_NL_ABB:>7.2f}    {Q0:>7.2f}  kVAR\n'
        f' k     {k_ABB:>7.5f}  {k:>9.5f}  kVAR/kW\u00b2\n'
        '──────────────────────────────────\n'
        ' P[kW]  PF_ABB  PF_ABB  PF_SCADA\n'
        '        meas.   params   params\n'
    )
    for P_pt, PF_ds in ABB_TEST:
        pf_t = _pf(P_pt, Q_NL_ABB, k_ABB)
        pf_f = _pf(P_pt, Q0, k)
        tbl += f' {P_pt:>5.1f}  {PF_ds:.4f}  {pf_t:.4f}  {pf_f:>7.4f}\n'
    tbl += ('──────────────────────────────────\n'
            f' RMS err        {rms_abb:.4f}  {rms_f4:>7.4f}')

    axL.text(
        0.97, 0.97, tbl,
        transform=axL.transAxes, ha='right', va='top',
        fontsize=7.5, family='monospace',
        bbox=dict(boxstyle='round,pad=0.5',
                  facecolor='#f4f7fb', edgecolor='#1F4E79',
                  linewidth=1.2, alpha=0.97),
        zorder=9,
    )

    # ----- RIGHT: Generator PF vs P, coloured per turbine -----
    if lbl is not None:
        unique_turbs = sorted(set(lbl.tolist()),
                              key=lambda x: int(x[1:]))
        cmap = plt.get_cmap('tab10')
        turb_colors = {t: cmap(i % 10) for i, t in enumerate(unique_turbs)}
        for tname in unique_turbs:
            m = (lbl == tname)
            axR.scatter(np.abs(P[m]), PF_meas[m], s=2, alpha=0.30,
                        c=[turb_colors[tname]],
                        label=f'{tname} (n={int(m.sum())})')
    else:
        axR.scatter(np.abs(P[off]), PF_meas[off], s=2, alpha=0.25, c='C3',
                    label='Cap OFF (measured)')
        axR.scatter(np.abs(P[on]), PF_meas[on], s=2, alpha=0.25, c='C0',
                    label='Cap ON (decompensated)')

    # Five binned-median points, positioned to interleave with the ABB
    # datasheet squares so the two marker sets don't overlap.
    # ABB P values: [1.04, 15.30, 29.52, 44.04, 58.25]
    # Median P values: midpoints between successive ABB points, plus one
    # above the highest ABB point.
    median_P_targets = [8.0, 22.5, 37.0, 51.0, 75.0]
    half_window = 2.0  # ± kW around each target
    centers, medians = [], []
    box_data = []  # for box-whisker overlay
    box_positions = []
    bin_stats = []  # descriptive statistics per bin
    for P_target in median_P_targets:
        mb = (np.abs(P) >= P_target - half_window) & \
             (np.abs(P) <= P_target + half_window)
        if mb.sum() >= 20:
            pf_bin = PF_meas[mb]
            centers.append(P_target)
            medians.append(float(np.median(pf_bin)))
            box_data.append(pf_bin)
            box_positions.append(P_target)
            q25, q75 = np.percentile(pf_bin, [25, 75])
            bin_stats.append({
                'n': int(mb.sum()),
                'med': float(np.median(pf_bin)),
                'iqr': float(q75 - q25),
                'std': float(np.std(pf_bin)),
                'q25': float(q25),
                'q75': float(q75),
            })
    if centers:
        axR.plot(centers, medians, 'D', ms=5,
                 markerfacecolor='#ff00ff', markeredgecolor='white',
                 markeredgewidth=1.0, linestyle='none',
                 label='Binned median', zorder=6)
    if box_data:
        bp = axR.boxplot(box_data, positions=box_positions, widths=3.0,
                         patch_artist=True, manage_ticks=False,
                         showfliers=False, whis=[0, 100], zorder=5,
                         medianprops=dict(color='#ff00ff', lw=1.5),
                         boxprops=dict(facecolor='#ff00ff', alpha=0.35,
                                       edgecolor='#ff00ff', lw=2.0),
                         whiskerprops=dict(color='#ff00ff', lw=1.5,
                                           linestyle='--'),
                         capprops=dict(color='#ff00ff', lw=2.0))
        # Annotate each box with descriptive stats, placed below the
        # lower whisker cap in a clear region away from the scatter.
        for i, (pc, st) in enumerate(zip(box_positions, bin_stats)):
            # Position: below the lower whisker cap
            whisk_lo = bp['whiskers'][2*i].get_ydata()[1]
            y_ann = whisk_lo - 0.06
            # Clamp to stay visible
            y_ann = max(y_ann, 0.03)
            ann_text = (f"P={pc:.0f} kW\n"
                        f"n={st['n']}\n"
                        f"med={st['med']:.3f}\n"
                        f"IQR={st['iqr']:.3f}\n"
                        f"σ={st['std']:.3f}")
            axR.text(pc, y_ann, ann_text, ha='center', va='top',
                     fontsize=5.5, family='monospace', color='#880088',
                     bbox=dict(boxstyle='round,pad=0.2',
                               facecolor='white', edgecolor='#cc88cc',
                               alpha=0.85, linewidth=0.5),
                     zorder=8)

    axR.plot(P_curve, PF_fit, 'k-', lw=2.5,
             label=f'Form 4 fit: Q\u2080={Q0:.3f}, k={k:.5f}')
    axR.plot(ABB_P, ABB_PF, 'rs', ms=10, mec='k', mew=1.2,
             label='ABB datasheet', zorder=7)

    axR.set_xlabel('P [kW]')
    axR.set_ylabel('Machine PF')
    axR.set_title(f'Machine PF: measured vs Form 4 '
                  f'(ABB err = {r["abb_err"]:.4f})', fontweight='bold')
    axR.set_xlim(0, 100)
    axR.set_ylim(0, 1.0)
    axR.legend(fontsize=8, loc='lower right', ncol=2)
    axR.grid(True, alpha=0.3)

    plt.suptitle(
        f"Machine Reactance aggregate calibration  |  N = {r['n_samples']:,}",
        fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    if out_path is None:
        out_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "form4_calibration.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if len(sys.argv) < 2:
        # Default: fleet calibration on T1-T6 (exclude T7)
        fleet = [
            os.path.join(_HERE, "10.0.103.10.csv"),  # T1
            os.path.join(_HERE, "10.1.78.11.csv"),   # T2
            os.path.join(_HERE, "10.0.153.10.csv"),  # T3
            os.path.join(_HERE, "10.0.182.10.csv"),  # T4
            os.path.join(_HERE, "10.1.20.10.csv"),   # T5
            os.path.join(_HERE, "10.1.27.10.csv"),   # T6
        ]
    else:
        fleet = sys.argv[1:]

    result = calibrate_form4(fleet)
    print_result(result)
    plot_path = plot_result(result)
    print(f"\n  Plot saved: {plot_path}")
