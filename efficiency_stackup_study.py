"""
efficiency_stackup_study.py
─────────────────────────────────────────────────────────────────────
Self-contained figure generator for the fleet-average loss stack-up
study (ABB M3AA 250SMA, 10-turbine fleet).

Compares a first-principles loss budget built from:
    (1) Stator Cu   — from the 1C1R thermal fit (steady-state T_ss)
    (2) Rotor Cu    — from the slip × P_elec identity
    (3) Iron+Fric+Wind — from the ABB no-load test (derived via a
                         heat balance at the stator winding node)

against the ABB datasheet's reported total losses (P₁ − P₂) at five
load points (NL, 25%, 50%, 75%, 100%).

All thermal-fit parameters below are the fleet averages across the
10-turbine Machine Study fleet. They are hardcoded here to keep this
script standalone. See provenance notes in PARAMETER_PROVENANCE below.

═════════════════════════════════════════════════════════════════════
PARAMETER_PROVENANCE
═════════════════════════════════════════════════════════════════════

1C1R THERMAL-FIT PARAMETERS (fleet mean across 10 machines):
    Rt_on_fleet = 0.04822    # °C/W
    P0_fleet    = 333.9      # W
  Source: run_fleet.py --plots
          (invokes turbine_thermal_pipeline_thevenin.fit_thevenin()
          on each machine; fleet means computed from the per-machine
          fit results and printed in the run summary)

ABB M3AA 250SMA TYPE TEST REPORT CONSTANTS:
    R_phase_20 = 0.09806 Ω   (per phase, delta, from cold resistance
                              test at 25.6°C reference ambient)
    alpha      = 0.00393 /°C (standard copper temp coefficient)
    s_rated    = 0.01333     (at 1480 rpm vs 1500 rpm synchronous)
  Source: ABB Type Test Report, "Resistance Measurements" and
          "Rated Performance" rows. R_LL_cold = 0.06681 Ω line-to-line;
          R_phase = (3/2)·R_LL for delta: 0.06681 × 1.5 = 0.10022,
          then normalized back to the 20°C reference:
          R_phase_20 = 0.10022 / (1 + 0.00393·5.6) = 0.09806 Ω.

ABB NO-LOAD TEST ROW:
    V_NL       = 400.8 V (line-to-line)
    I_NL_line  = 35.6 A
    P_NL       = 1040 W  (three-phase total at no-load)
    cos φ_NL   = 0.04
  Source: ABB Type Test Report, "No-Load Test" row.

ABB LOAD POINTS (from datasheet "Rated Performance" and partial-load
points tables):
    NL:   P1=1.04 kW, P2=0,  I=N/A, η=0,      (from P_NL above)
    25%:  P1=15.30,   P2=13.75, I=43.4, η=0.8987, PF=0.51
    50%:  P1=29.52,   P2=27.50, I=58.8, η=0.9316, PF=0.72
    75%:  P1=44.04,   P2=41.25, I=78.5, η=0.9366, PF=0.81
    100%: P1=58.25,   P2=55.00, I=99.5, η=0.9442, PF=0.84

Output: efficiency_stackup_study.png in the same directory as the script.

Usage:
    python3 efficiency_stackup_study.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ═════════════════════════════════════════════════════════════════════
# HARDCODED FLEET-AVERAGE PARAMETERS
# ═════════════════════════════════════════════════════════════════════

# Fleet-mean thermal-fit parameters (from run_fleet.py)
Rt_on_fleet = 0.04822    # °C/W
P0_fleet    = 333.9      # W

# ABB Type Test Report constants
R_phase_20  = 0.09806    # Ω, per phase (delta), at 20°C reference
alpha       = 0.00393    # /°C, copper temperature coefficient
s_rated     = 0.01333    # rated slip
T_amb_test  = 25.6       # °C, ABB cold-resistance test ambient
# Note: T_amb = 25.6°C is used throughout this study — for the no-load
# row AND the loaded rows. The rationale is that the ABB Type Test Report
# reports the cold-resistance ambient as 25.6°C and the thermal test ambient
# as 26°C (essentially the same), so a single consistent value is used.
# See PARAMETER_PROVENANCE in module docstring.
T_amb_op    = 25.6       # °C, operating ambient (unified with test ambient)

# ABB no-load test row
I_NL_line   = 35.6       # A
P_NL        = 1040.0     # W

# ABB load points: (label, P1_kW, P2_kW, I_line_A)
ABB_POINTS = [
    # label,  P1_kW, P2_kW, I_line_A (all from ABB Type Test Report,
    # Partial Load Points + Thermal test row, verified by direct
    # read of the datasheet on 2026-04-12)
    ('NL*',    1.04,   0.00,  35.6),    # no-load test row
    ('25%',   15.30,  14.13,  43.4),    # η=92.34%, n=1496 rpm
    ('50%',   29.52,  27.94,  58.8),    # η=94.66%, n=1491 rpm
    ('75%',   44.04,  41.79,  78.5),    # η=94.85%, n=1486 rpm
    ('100%',  58.25,  55.00,  99.5),    # η=94.42%, n=1480 rpm (thermal test)
]

# Rotor temperature offset above stator winding (no longer used in
# computation — kept here for documentation of the prior assumption).
# See PARAMETER_PROVENANCE in module docstring for the rationale on
# why temperature correction was removed.
DELTA_T_ROTOR = 0.0      # °C  (unused)


# ═════════════════════════════════════════════════════════════════════
# PHYSICS HELPERS
# ═════════════════════════════════════════════════════════════════════

def R1_at(T_celsius):
    """Stator winding phase resistance at temperature T (°C)."""
    return R_phase_20 * (1 + alpha * (T_celsius - 20))


def solve_Tss(I_line, T_amb, Rt_on, P0):
    """
    Steady-state winding temperature from the 1C1R heat balance:
        (T_ss - T_amb)/Rt_on = 3·I²·R_phase(T_ss) + P0

    Solved algebraically (closed-form since R is linear in T).
    Returns T_ss in °C.
    """
    I_ph = I_line / np.sqrt(3)        # phase current (delta)
    K = 3 * I_ph**2 * R_phase_20       # W at 20°C
    num = T_amb + Rt_on * (K * (1 - alpha * 20) + P0)
    den = 1.0 - Rt_on * K * alpha
    return num / den


def stator_Cu_at(I_line, T_ss):
    """Stator copper loss P_Cu = 3·I_ph²·R₁(T_ss), watts."""
    I_ph = I_line / np.sqrt(3)
    return 3 * I_ph**2 * R1_at(T_ss)


def rotor_Cu_at(P1_kW, T_ss):
    """
    Rotor copper loss via slip × P_elec identity.
    Slip scales linearly with P1 from the rated-point anchor.
    No temperature correction — the slip identity is already an
    operating-condition statement (the measured slip reflects whatever
    rotor temperature obtained at thermal equilibrium during the ABB test).
    The T_ss argument is unused but retained for API compatibility.
    """
    s = s_rated * (P1_kW / 58.25)      # linear-in-P1 slip
    return s * P1_kW * 1000.0          # watts


# ═════════════════════════════════════════════════════════════════════
# DERIVED: P_ifw FROM THE NO-LOAD HEAT BALANCE
# ═════════════════════════════════════════════════════════════════════

# Solve for winding temperature at no-load equilibrium
T_w_NL = solve_Tss(I_NL_line, T_amb_test, Rt_on_fleet, P0_fleet)
P_Cu_NL_refined = stator_Cu_at(I_NL_line, T_w_NL)
P_ifw = P_NL - P_Cu_NL_refined
P_ifw_remainder = P_ifw - P0_fleet


# ═════════════════════════════════════════════════════════════════════
# LOSS BUDGET AT EACH LOAD POINT
# ═════════════════════════════════════════════════════════════════════

labels = []
P1_vals = []
P_Cu_s_vals = []
P_rCu_vals = []
P_ABB_vals = []
P_pred_3term_vals = []     # 3-term prediction (before stray fit)
delta_3term_vals = []      # residual after 3-term budget

for lbl, P1_kW, P2_kW, I_line in ABB_POINTS:
    labels.append(lbl)
    P1_vals.append(P1_kW)
    P_ABB_tot = (P1_kW - P2_kW) * 1000.0  # W

    if lbl.startswith('NL'):
        # NL row anchors the calculation — use the directly-computed value
        P_Cu_s = P_Cu_NL_refined
        P_rCu = 0.0
        P_pred = P_Cu_s + 0 + P_ifw       # equals P_NL by construction
        P_ABB_tot = P_NL
    else:
        T_ss = solve_Tss(I_line, T_amb_op, Rt_on_fleet, P0_fleet)
        P_Cu_s = stator_Cu_at(I_line, T_ss)
        P_rCu = rotor_Cu_at(P1_kW, T_ss)
        P_pred = P_Cu_s + P_rCu + P_ifw

    P_Cu_s_vals.append(P_Cu_s)
    P_rCu_vals.append(P_rCu)
    P_ABB_vals.append(P_ABB_tot)
    P_pred_3term_vals.append(P_pred)
    delta_3term_vals.append(P_ABB_tot - P_pred)

# ═════════════════════════════════════════════════════════════════════
# STRAY LOAD LOSS FIT (IEEE 112 residual method)
# ═════════════════════════════════════════════════════════════════════
# Fit P_stray = c · (P1/P_rated)² · P_rated to the 3-term residual Δ
# using the four loaded rows only (NL is excluded: zero by construction).

P_rated_W = 58250.0
P1_arr = np.array([p * 1000 for p in P1_vals])  # W
delta_arr = np.array(delta_3term_vals)

# Design matrix for quadratic-in-P1 fit (forced through origin)
# f(P1) = c · (P1/P_rated)² · P_rated
loaded_mask = np.array([lbl != 'NL*' for lbl in labels])
x_fit = (P1_arr[loaded_mask] / P_rated_W) ** 2 * P_rated_W
y_fit = delta_arr[loaded_mask]
c_stray = float(np.sum(x_fit * y_fit) / np.sum(x_fit * x_fit))

# Compute P_stray at each load point and the final 4-term prediction
P_stray_vals = [c_stray * (p1 * 1000 / P_rated_W) ** 2 * P_rated_W
                for p1 in P1_vals]
P_pred_vals = [p3 + ps for p3, ps in zip(P_pred_3term_vals, P_stray_vals)]
delta_vals = [int(round(pa - pp))
              for pa, pp in zip(P_ABB_vals, P_pred_vals)]


# ═════════════════════════════════════════════════════════════════════
# FIGURE
# ═════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(15, 8.8))
gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.10)

# LEFT: stacked bar chart
ax = fig.add_subplot(gs[0, 0])
x = np.arange(len(labels))
bar_w = 0.55
c_stator = '#CC6677'
c_rotor  = '#DDCC77'
c_ifw_p0 = '#3377AA'
c_ifw_rm = '#88CCEE'

ax.bar(x, P_Cu_s_vals, bar_w, color=c_stator, edgecolor='white',
       linewidth=1.0,
       label='Stator Cu  =  I²·R_dc(T_ss)   [thermal fit]')
ax.bar(x, P_rCu_vals, bar_w, bottom=P_Cu_s_vals, color=c_rotor,
       edgecolor='white', linewidth=1.0,
       label='Rotor Cu  =  s·P_elec   [slip identity]')

bot3 = [a + b for a, b in zip(P_Cu_s_vals, P_rCu_vals)]
ax.bar(x, [P0_fleet] * len(x), bar_w, bottom=bot3, color=c_ifw_p0,
       edgecolor='white', linewidth=1.0,
       label=f'  └─ P₀ ≈ {P0_fleet:.0f} W  (winding-coupled fraction)')

bot4 = [a + P0_fleet for a in bot3]
ax.bar(x, [P_ifw_remainder] * len(x), bar_w, bottom=bot4, color=c_ifw_rm,
       edgecolor='white', linewidth=1.0,
       label=f'Iron+Fric+Wind   (remainder ≈ {P_ifw_remainder:.0f} W '
             '→ frame/shaft)')

# Stray load loss (4th component, topmost slice)
bot5 = [a + P_ifw_remainder for a in bot4]
c_stray_color = '#AA4499'  # purple
ax.bar(x, P_stray_vals, bar_w, bottom=bot5, color=c_stray_color,
       edgecolor='white', linewidth=1.0,
       label=f'Stray load loss  =  '
             f'{c_stray*58250:.0f} W·(P₁/P₁,rated)²   [IEEE 112]')

# Δ annotations above each bar (ABB horizontal markers removed —
# the bars now match the ABB total directly within ±13 W)
for i, xi in enumerate(x):
    color = '#CC1B1B' if delta_vals[i] > 0 else '#117733'
    ax.annotate(f'Δ={delta_vals[i]:+d} W',
                xy=(xi, P_ABB_vals[i]),
                xytext=(xi, P_ABB_vals[i] + 130), ha='center',
                fontsize=9, fontweight='bold', color=color)

# Total prediction labels — placed above the bar top on every bar
# (including NL now that the 580 W remainder label no longer clashes
# with a red ABB horizontal)
for i, xi in enumerate(x):
    y_top = P_pred_vals[i]
    ax.text(xi, y_top + 35, f'{P_pred_vals[i]:.0f} W',
            ha='center', va='bottom',
            fontsize=8.5, fontweight='bold', color='#1F4E79')

# Iron+fric+wind remainder (580 W) labels — inside the light-blue slice,
# on every bar including NL
for i, xi in enumerate(x):
    y_remainder = bot4[i] + P_ifw_remainder / 2
    ax.text(xi, y_remainder, f'{P_ifw_remainder:.0f} W',
            ha='center', va='center',
            fontsize=8, fontweight='bold', color='#1F4E79')

# Stator Cu numeric labels — inside the red slice on every bar
for i, xi in enumerate(x):
    y_stator = P_Cu_s_vals[i] / 2
    ax.text(xi, y_stator, f'{P_Cu_s_vals[i]:.0f} W',
            ha='center', va='center',
            fontsize=8, fontweight='bold', color='white')

# Rotor Cu numeric labels — inside the yellow slice, only for 50/75/100%
for i, xi in enumerate(x):
    if i >= 2 and P_rCu_vals[i] > 0:
        y_rotor = P_Cu_s_vals[i] + P_rCu_vals[i] / 2
        ax.text(xi, y_rotor, f'{P_rCu_vals[i]:.0f} W',
                ha='center', va='center',
                fontsize=8, fontweight='bold', color='#555500')

# P₀ labels inside the dark-blue slice (numeric only, no prefix)
for i, xi in enumerate(x):
    y_p0 = bot3[i] + P0_fleet / 2
    ax.text(xi, y_p0, f'{P0_fleet:.0f} W',
            ha='center', va='center',
            fontsize=7.5, fontweight='bold', color='white')

# P₁ labels under x-axis
for xi, p1, lbl in zip(x, P1_vals, labels):
    ax.text(xi, -180, f'{lbl}\nP₁={p1:.1f} kW',
            ha='center', va='top', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels([])
ax.set_ylabel('Power loss  [W]', fontsize=11)
ax.set_title('Fleet-average loss stack-up at ABB load points',
             fontweight='bold', fontsize=12)
ax.set_ylim(0, 4100)
ax.set_xlim(-0.65, len(labels) - 0.35)
ax.grid(True, axis='y', alpha=0.3)
# Build legend with Remainder placed before P0 (parent → child)
handles, labels_leg = ax.get_legend_handles_labels()
# Bar order: [Stator, Rotor, P0, Remainder, Stray]
# Swap P0 and Remainder so parent sits above child (└─ P0):
# New order: [Stator, Rotor, Remainder, P0, Stray]
order = [0, 1, 3, 2, 4]
ax.legend([handles[i] for i in order], [labels_leg[i] for i in order],
          loc='upper left', fontsize=8, framealpha=0.95)

# RIGHT: equations + result table
gs_r = gs[0, 1].subgridspec(2, 1, height_ratios=[3.4, 0.9], hspace=0.05)
ax_eq = fig.add_subplot(gs_r[0, 0])
ax_tb = fig.add_subplot(gs_r[1, 0])
for a in (ax_eq, ax_tb):
    a.axis('off')
    a.set_xlim(0, 1)
    a.set_ylim(0, 1)

eqns = [
    (0.98, r'$\bf{Loss\ decomposition\ equations}$', 12),
    (0.92, r'$\bf{1.\ Stator\ Cu\ }$(1C1R thermal fit):', 10),
    (0.87, r'$P_{Cu,s} = I^2 R_{dc}(T_{ss}),\ \ '
           r'R_{dc}(T) = R_{dc,20}[1{+}\alpha(T{-}20)]$', 11),
    (0.81, r'$\bf{2.\ Rotor\ Cu\ }$(slip identity, no temp correction):', 10),
    (0.75, r'$P_{rCu} = s\cdot P_{elec}\ \ '
           r'(\mathrm{energy\ conservation\ at\ air\ gap})$', 10),
    (0.69, r'$\bf{2a.\ Slip\ linearity\ }$(linear $s$–$P_1$):', 10),
    (0.63, r'$s(P_1) = s_{rated}\cdot P_1/P_{1,rated},\ \ '
           r's_{rated}=0.01333,\ \ P_{1,rated}=58.25\,\mathrm{kW}$', 10),
    (0.55, r'$\bf{3.\ Iron+Fric+Wind\ }$(ABB no-load test):', 10),
    (0.49, r'$P_{ifw} = P_{NL} - 3\,I_{NL}^2\,R_1(T_{w,NL})$', 11),
    (0.42, r'$(T_{w,NL}-T_{amb})/R_{t,on} = 3 I_{NL}^2 R_1(T_{w,NL}) + P_0$',
           10),
    (0.35, rf'$\Rightarrow T_{{w,NL}} = {T_w_NL:.1f}\,°C,\ '
           rf'P_{{ifw}} = {P_NL:.0f}{{-}}{P_Cu_NL_refined:.0f} = '
           rf'\bf{{{P_ifw:.0f}\ W}}$', 11),
    (0.28, r'$\mathrm{note:}\ P_0 \subset P_{ifw}\ '
           r'(\mathrm{winding\text{-}coupled\ fraction\ shown\ in\ dark\ blue})$',
           8.5),
    (0.20, r'$\bf{4.\ Stray\ Load\ Loss\ }$(IEEE 112 §5.3, '
           rf'ABB "PLL from residual"):',
           9),
    (0.13, rf'$P_{{stray}}(P_1) = P_{{stray,rated}}\cdot'
           rf'(P_1/P_{{1,rated}})^2,\ \ '
           rf'P_{{stray,rated}}={c_stray*58250:.0f}\,\mathrm{{W}}$', 10),
    (0.07, rf'(equivalent to ${100*c_stray:.3f}\%$ of rated input; '
           r'fitted from 3-term residual)', 8),
    (0.00, r'$\bf{Prediction:}\ P_{pred} = P_{Cu,s} + P_{rCu} + P_{ifw} '
           r'+ P_{stray}$', 10),
]
for y, t, sz in eqns:
    ax_eq.text(0.02, y, t, fontsize=sz, transform=ax_eq.transAxes, va='top')

tbl_text = (
    r'$\bf{Result\ summary}$' + '\n'
    '──────────────────────────────────────────\n'
    f"{'Load':<6}{'Pred':>8}{'ABB (P₁−P₂)':>18}{'Δ':>9}\n"
    '──────────────────────────────────────────\n'
)
for lbl, pp, pa, d, p1, p2 in zip(labels, P_pred_vals, P_ABB_vals,
                                    delta_vals, P1_vals,
                                    [0.0, 14.13, 27.94, 41.79, 55.00]):
    if lbl.startswith('NL'):
        # NL row: from the ABB No-Load test row, not Partial Load Points.
        # P_NL is input power at zero shaft load.
        abb_str = f'P_NL = 1.04 kW'
    else:
        abb_str = f'{p1:.2f}−{p2:.2f}={pa/1000:.2f} kW'
    tbl_text += f'{lbl:<6}{pp:>6.0f} W  {abb_str:>20}{d:>+6d} W\n'
tbl_text += (
    '──────────────────────────────────────────\n'
    '* NL row: anchor (closes by construction)\n'
    '  P_NL from No-Load test; P₁, P₂ from\n'
    '  Partial Load Points (ABB Type Test Report)'
)
ax_tb.text(0.02, 0.95, tbl_text, fontsize=7.5, family='monospace',
           transform=ax_tb.transAxes, va='top',
           bbox=dict(boxstyle='round,pad=0.4',
                     facecolor='#f4f7fb', edgecolor='#1F4E79',
                     linewidth=1.2))

plt.suptitle(
    'Efficiency Budget: Fleet-Average Loss Stack-Up vs ABB Datasheet',
    fontsize=13, fontweight='bold', y=0.99)

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'efficiency_stackup_study.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)

# ═════════════════════════════════════════════════════════════════════
# CONSOLE OUTPUT (sanity check)
# ═════════════════════════════════════════════════════════════════════

print("=" * 72)
print("  Efficiency stack-up study — self-contained regeneration")
print("=" * 72)
print(f"\n  Winding temp at no-load equilibrium: T_w,NL = {T_w_NL:.1f} °C")
print(f"  Refined no-load stator Cu:           {P_Cu_NL_refined:.1f} W")
print(f"  Derived P_ifw:                       {P_ifw:.1f} W")
print(f"  (of which P₀ = {P0_fleet:.0f} W is winding-coupled,")
print(f"   remainder = {P_ifw_remainder:.0f} W heats frame/shaft)")
print()
print(f"  {'Load':<6}{'P1[kW]':>8}{'P_Cu_s':>9}{'P_rCu':>9}"
      f"{'Pred':>9}{'ABB':>9}{'Δ':>9}")
print("  " + "─" * 57)
for lbl, p1, cs, rc, pp, pa, d in zip(labels, P1_vals, P_Cu_s_vals,
                                       P_rCu_vals, P_pred_vals,
                                       P_ABB_vals, delta_vals):
    print(f"  {lbl:<6}{p1:>7.2f}{cs:>8.0f}W{rc:>7.0f}W"
          f"{pp:>7.0f}W{pa:>7.0f}W{d:>+7d}W")
print("  " + "─" * 57)

mean_abs_loaded = np.mean([abs(d) for lbl, d in zip(labels, delta_vals)
                            if lbl != 'NL*'])
print(f"\n  Mean |Δ| at loaded points (25/50/75/100%): {mean_abs_loaded:.0f} W")
print(f"  As fraction of rated P1:                   "
      f"{100*mean_abs_loaded/58250:.2f}%")
print(f"\n  Figure saved: {out_path}")
