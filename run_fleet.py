"""
Two-stage fleet driver.

Stage 1 — Form 4 calibration:
    Run `calibrate_form4()` on a list of SCADA CSVs to produce the four
    fleet constants (Q0, k, Q_cap, V_nom) from the raw data.

Stage 2 — Per-turbine Thévenin (1C1R) thermal identification:
    For each turbine CSV, feed the calibrated (Q0, k) into the Thévenin
    pipeline so I_gen is computed with the freshly-derived Form 4
    instead of the hardcoded module defaults. Four parameters are
    fitted per turbine: Rt_on, Rt_off, P0, C_eq.

Usage:
    python run_fleet.py                      # default fleet, no plots
    python run_fleet.py --plots              # default fleet + standard plots
    python run_fleet.py --plots --reporting  # full pipeline + report PDF
    python run_fleet.py path1.csv path2.csv  # custom fleet

Flags:
    --plots      Generate per-turbine and fleet diagnostic plots.
    --reporting  Chain efficiency_stackup_study.py and generate_report_pdf.py
                 after the main pipeline to produce the final report PDF.
                 Implies --plots (plots are required for the full report).

Plots produced when --plots is set:
    form4_calibration.png       — Q vs P scatter + PF vs P with ABB anchors
    <label>_thevenin_summary.png — six-panel per-turbine fit summary,
                                   one file per turbine in the fleet
"""

import sys
import os
import numpy as np

# Ensure both companion modules are importable when this script is run
# directly from its install directory.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import form4_calibration as f4
import turbine_thermal_pipeline_thevenin as pl


# Resolve CSV paths relative to this script's directory, so the scripts
# work wherever the repo is checked out.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _csv(filename):
    return os.path.join(_SCRIPT_DIR, filename)


# ----- default fleet (all 8 turbines used for both calibration and run) -----

DEFAULT_FLEET = [
    ("T1", _csv("10.0.103.10.csv")),
    ("T2", _csv("10.1.78.11.csv")),
    ("T3", _csv("10.0.153.10.csv")),
    ("T4", _csv("10.0.182.10.csv")),
    ("T5", _csv("10.1.20.10.csv")),
    ("T6", _csv("10.1.27.10.csv")),
    ("T7", _csv("10.1.111.10.csv")),
    ("T8", _csv("10.1.179.10.csv")),
    ("T9", _csv("10.1.166.10.csv")),
    ("T10", _csv("10.1.181.10.csv")),
]

# All turbines now contribute to the fleet Form 4 calibration. Earlier
# revisions excluded T7 because it was a recent addition with only the
# heavy-duty operating point; that exclusion was lifted once we verified
# T7 is consistent with the rest of the fleet and adding it pulled the
# Form 4 fit closer to the ABB datasheet (ABB err 0.0042 -> 0.0026).
# T8 was added under the same justification.
CALIBRATION_TURBINES = ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10")


# ----- stage-2 replica of pipeline.process() with explicit Form 4 override -----

def process_with_form4(path, form4):
    """
    Replica of `turbine_thermal_pipeline_thevenin.process()` that passes
    (Q0, k) from a calibration dict into `compute_I_gen` explicitly, instead
    of relying on the module-level defaults.
    """
    Q0 = form4["Q0"]
    k = form4["k"]

    data = pl.load_scada(path)
    N = data["N"]

    # 1) Cap state detection (uses the same refractory filter)
    cap_on = pl.detect_cap_state(data["PF_meter_raw"])

    # 2) Generator current via Form 4  --  OVERRIDE HERE
    I_gen = pl.compute_I_gen(data["P"], data["V"], Q0=Q0, k=k)
    I2 = I_gen ** 2
    I2[~data["gen_on"]] = 0

    # 3) Thermal fit
    fit = pl.fit_thevenin(N, data["dt"], data["Tw"], data["Tnac"], I2,
                          data["gen_on"], data["gap"])

    # 4) Innovation diagnostics
    nu, valid = pl.extract_innovation_thevenin(
        fit["Rt_on"], fit["Rt_off"], fit["P0"], fit["C_eq"],
        N, data["dt"], data["Tw"], fit["Tw_pred"], data["Tnac"],
        I2, data["gen_on"], data["gap"],
    )
    diag = pl.innovation_diagnostics(nu, valid, data["gen_on"], I2)

    # 5) Damage curve and continuous rating
    damage = {
        m: pl.trip_time(fit["Rt_on"], fit["P0"], fit["C_eq"], 40.0, m)
        for m in [1.1, 1.2, 1.5, 2.0, 3.0, 5.0]
    }
    I_cont = pl.continuous_current_rating(fit["Rt_on"], fit["P0"], fit["C_eq"])

    return {
        "data": data,
        "cap_on": cap_on,
        "I_gen": I_gen,
        "fit": fit,
        "innovation": nu,
        "innovation_valid": valid,
        "stochastic": diag,
        "damage_curve": damage,
        "I_continuous_mult": I_cont,
        "form4_used": {"Q0": Q0, "k": k},
    }


def _run_thevenin(fleet, form4, plot_dir=None):
    """
    Run the Thévenin (1C1R) pipeline for every turbine in the fleet.

    plot_dir : str or None
        If provided, write a six-panel summary plot per turbine into this
        directory. File names are `<plot_dir>/<label>_thevenin_summary.png`.
        Created if missing.
    """
    print("\n" + "=" * 70)
    print("  STAGE 2 -- Thevenin (1C1R) thermal identification")
    print("=" * 70)
    print(f"  Q0 = {form4['Q0']:.4f}, k = {form4['k']:.6f}  (from Stage 1)\n")

    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)

    rows = []
    for (label, path) in fleet:
        result = process_with_form4(path, form4)
        fit = result["fit"]
        diag = result["stochastic"]
        I_cont = result["I_continuous_mult"]
        acf1 = diag["acf"][0] if diag["acf"] else float("nan")
        gm = result["data"]["gen_on"] & (result["data"]["P"] > 1)
        I_mean = float(result["I_gen"][gm].mean()) if gm.any() else float("nan")
        rows.append({
            "label": label, "rmse": fit["rmse"], "acf1": acf1,
            "Rt_on": fit["Rt_on"], "P0": fit["P0"], "C_eq": fit["C_eq"],
            "tau_on_min": fit["tau_on"] / 60,
            "I_cont": I_cont if I_cont is not None else float("nan"),
            "I_mean": I_mean,
        })

        if plot_dir is not None:
            out = os.path.join(plot_dir, f"{label}_thevenin_summary.png")
            file_stem = os.path.splitext(os.path.basename(path))[0]
            fig_title = f"{file_stem}  —  Machine Study"
            pl.plot_summary(result, label=label, out_path=out,
                            title=fig_title)

    print(f"  {'Turb':<5} {'RMSE':>7} {'ACF(1)':>8} "
          f"{'Rt_on':>9} {'P0':>7} {'C_eq':>8} "
          f"{'tau_on':>9} {'Ic/In':>8} {'I_mean[A]':>9}")
    print("  " + "-" * 80)
    for r in rows:
        print(f"  {r['label']:<5} "
              f"{r['rmse']:>7.3f} {r['acf1']:>+8.4f} "
              f"{r['Rt_on']:>9.5f} {r['P0']:>7.0f} {r['C_eq']:>8.0f} "
              f"{r['tau_on_min']:>7.1f}m {r['I_cont']:>8.3f} {r['I_mean']:>9.2f}")

    mean_rmse = float(np.mean([r["rmse"] for r in rows]))
    print(f"\n  Thévenin mean RMSE: {mean_rmse:.4f} °C")
    if plot_dir is not None:
        print(f"  Per-turbine plots written to: {plot_dir}/")
    return rows


# ----- main ---------------------------------------------------------------

def main(fleet=None, make_plots=False, plot_dir=None):
    """
    Run the fleet driver.

    Parameters
    ----------
    fleet : list of (label, path) tuples, optional
        Defaults to DEFAULT_FLEET.
    make_plots : bool
        If True, generate the standard pipeline plots: a Form 4
        calibration summary, a ZEV relay trip curves reference sheet
        (both fleet-wide, one figure each), and a six-panel Thévenin
        fit summary per turbine. Default False.
    plot_dir : str, optional
        Directory for plot outputs. Defaults to the script's own
        directory if make_plots is True and plot_dir is None.
    """
    if fleet is None:
        fleet = DEFAULT_FLEET

    if make_plots and plot_dir is None:
        plot_dir = _SCRIPT_DIR

    cal_paths = [p for (lbl, p) in fleet if lbl in CALIBRATION_TURBINES]

    # ---- Stage 1: Form 4 calibration ------------------------------------
    print("=" * 70)
    print("  STAGE 1 -- Form 4 fleet calibration")
    print("=" * 70)
    form4 = f4.calibrate_form4(cal_paths)
    f4.print_result(form4)

    if make_plots:
        f4_path = os.path.join(plot_dir, "form4_calibration.png")
        f4.plot_result(form4, out_path=f4_path)
        print(f"\n  Form 4 calibration plot: {f4_path}")

        zev_path = os.path.join(plot_dir, "zev_trip_curves.png")
        pl.plot_zev_curves(out_path=zev_path)
        print(f"  ZEV trip curves plot:    {zev_path}")

        eqn_path = os.path.join(plot_dir, "model_equations.png")
        pl.plot_equations_page(out_path=eqn_path)
        print(f"  Model equations page:    {eqn_path}")

    # ---- Stage 2: Thévenin thermal identification ----------------------
    thev_rows = _run_thevenin(fleet, form4,
                                plot_dir=plot_dir if make_plots else None)

    return {
        "form4": form4,
        "thevenin": thev_rows,
    }


if __name__ == "__main__":
    # Positional args are CSV paths; --plots enables standard pipeline
    # plots (Form 4 calibration + per-turbine Thévenin summary).
    # --reporting chains efficiency_stackup_study.py and
    # generate_report_pdf.py after the main pipeline.
    args = sys.argv[1:]
    make_plots = False
    make_report = False
    paths = []
    for a in args:
        if a == "--plots":
            make_plots = True
        elif a == "--reporting":
            make_report = True
        elif a.startswith("--"):
            print(f"Warning: flag {a} is not recognised in this revision.")
        else:
            paths.append(a)

    if paths:
        custom = [(f"T{i + 1}", p) for i, p in enumerate(paths)]
        main(custom, make_plots=make_plots or make_report)
    else:
        main(make_plots=make_plots or make_report)

    if make_report:
        import subprocess
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for script in ["efficiency_stackup_study.py",
                        "generate_report_pdf.py"]:
            script_path = os.path.join(script_dir, script)
            print(f"\n  Running {script} ...")
            subprocess.run([sys.executable, script_path],
                           cwd=script_dir, check=True)
