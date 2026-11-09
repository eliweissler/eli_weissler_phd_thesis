"""Transmon numerics figure for chapter 0.

Runs ../../code-snippets/chapter_0.py (the source of the chapter's code listings) and plots the
results; writes transmon_convergence.pdf next to this file.
"""
import os
import runpy
import numpy as np
from _convergence_plot import *

here = os.path.dirname(os.path.abspath(__file__))
ns = runpy.run_path(os.path.join(here, "..", "..", "code-snippets", "chapter_0.py"))
EJ, EC = 5, 0.2  # parameters used in the transmon listings

fig, (ax_wf, ax_disp, ax_basis, ax_grid) = new_figure()

# (a) Wavefunctions on the periodic phase grid at n_g = 0.
phi_vals, evals, evecs = ns["transmon_phase_fd"](160, EJ, EC, 0, 8)
dphi = phi_vals[1] - phi_vals[0]
wavefunction_panel(ax_wf, phi_vals, -EJ*np.cos(phi_vals), evals,
                   [evecs[:, i]/np.sqrt(dphi) for i in range(3)], r"$\phi$", ylim=(-5.5, 7.5))
ax_wf.set_xlim(-np.pi, np.pi)
ax_wf.set_xticks([-np.pi, 0, np.pi])
ax_wf.set_xticklabels([r"$-\pi$", r"$0$", r"$\pi$"])

# (b) Lowest four levels vs. offset charge (charge basis).
dispersion_panel(ax_disp, ns["ng_vals"], ns["levels"], r"$n_g$")

# (c) Charge basis convergence.
charge_err = ns["charge_err"]
basis_panel(ax_basis, np.array(list(charge_err)), np.array(list(charge_err.values())), r"charge cutoff $n_{\rm cut}$")

# (d) Periodic phase grid convergence for different stencil orders.
grid_panel(ax_grid, ns["grid_err_tr"])

out = os.path.join(here, "transmon_convergence.pdf")
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
