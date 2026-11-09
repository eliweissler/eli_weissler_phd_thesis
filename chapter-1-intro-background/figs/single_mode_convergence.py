"""Combined fluxonium and transmon numerics figure for the numerical-modeling section.

Rows: fluxonium (top) and transmon (bottom).
Columns: wavefunctions, basis convergence, phase-grid convergence.
Runs ../../code-snippets/chapter_0.py (the source of the chapter's code listings) and plots the
results; writes single_mode_convergence.pdf next to this file.
"""
import os
import runpy
import numpy as np
import matplotlib.ticker
from _convergence_plot import *

here = os.path.dirname(os.path.abspath(__file__))
ns = runpy.run_path(os.path.join(here, "..", "..", "code-snippets", "chapter_0.py"))

plt.rcParams.update({"font.size": 8, "mathtext.fontset": "cm", "font.family": "serif", "axes.linewidth": 0.6})
fig = plt.figure(figsize=(6.5, 4.0))
gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.62)
axes = [[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(2)]
for row, name in zip(axes, ["fluxonium", "transmon"]):
    for ax, tag in zip(row, "abc" if name == "fluxonium" else "def"):
        ax.set_title(f"({tag}) {name}", loc="left", fontsize=8)

# --- Fluxonium ---
EJ, EC, EL, phi_ext = ns["EJ"], ns["EC"], ns["EL"], ns["phi_ext"]
ax_wf, ax_basis, ax_grid = axes[0]

# (a) Output of the phase-basis listing (default range [-5pi, 3pi], 201 points).
phi_vals, evals, evecs = ns["fluxonium_phase"](201, EJ, EC, EL, phi_ext)
wavefunction_panel(ax_wf, phi_vals, 0.5*EL*(phi_vals + phi_ext)**2 - EJ*np.cos(phi_vals), evals,
                   [10*evecs[:, i] for i in range(N_STATES)], r"$\phi$", ylim=(-10, 20),
                   legend_kw={"ncol": 2, "fontsize": 5.5})
ax_wf.set_xlim(-3*np.pi, np.pi)
ax_wf.set_xticks(np.pi*np.arange(-3, 2))
ax_wf.set_xticklabels([r"$-3\pi$", r"$-2\pi$", r"$-\pi$", r"$0$", r"$\pi$"])

# (b) Fock basis convergence.
fock_err = ns["fock_err"]
basis_panel(ax_basis, np.array(list(fock_err)), np.array(list(fock_err.values())), "Fock levels")
ax_basis.set_xticks([10, 40, 70, 100])

# (c) Phase grid convergence for different stencil orders.
grid_panel(ax_grid, ns["grid_err"])

# --- Transmon ---
EJ, EC = 5, 0.2  # parameters used in the transmon listings
ax_wf, ax_basis, ax_grid = axes[1]

# (d) Wavefunctions on the periodic phase grid at n_g = 0.
phi_vals, evals, evecs = ns["transmon_phase_fd"](160, EJ, EC, 0, 8)
dphi = phi_vals[1] - phi_vals[0]
wavefunction_panel(ax_wf, phi_vals, -EJ*np.cos(phi_vals), evals,
                   [evecs[:, i]/np.sqrt(dphi) for i in range(3)], r"$\phi$", ylim=(-5.5, 7.5),
                   legend_kw={"ncol": 2, "fontsize": 5.5})
ax_wf.set_xlim(-np.pi, np.pi)
ax_wf.set_xticks([-np.pi, 0, np.pi])
ax_wf.set_xticklabels([r"$-\pi$", r"$0$", r"$\pi$"])

# (e) Charge basis convergence.
charge_err = ns["charge_err"]
basis_panel(ax_basis, np.array(list(charge_err)), np.array(list(charge_err.values())), r"charge cutoff $n_{\rm cut}$")

# (f) Periodic phase grid convergence for different stencil orders.
grid_panel(ax_grid, ns["grid_err_tr"])

# Smaller panels: thin the log-axis labels and legends.
for row in axes:
    for ax in row[1:]:
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=6))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.legend(frameon=False, fontsize=5.5, loc="lower right" if ax is row[2] else "upper right", handlelength=1.2)

out = os.path.join(here, "single_mode_convergence.pdf")
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
