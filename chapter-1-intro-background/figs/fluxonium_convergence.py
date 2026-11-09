"""Fluxonium numerics figure for the Convergence subsection of chapter 0.

Runs ../../code-snippets/chapter_0.py (the source of the chapter's code listings) and plots the
results; writes fluxonium_convergence.pdf next to this file.
"""
import os
import runpy
import numpy as np
from _convergence_plot import *

here = os.path.dirname(os.path.abspath(__file__))
ns = runpy.run_path(os.path.join(here, "..", "..", "code-snippets", "chapter_0.py"))
EJ, EC, EL, phi_ext = ns["EJ"], ns["EC"], ns["EL"], ns["phi_ext"]

fig, (ax_wf, ax_disp, ax_basis, ax_grid) = new_figure()

# (a) Output of the phase-basis listing (default range [-5pi, 3pi], 201 points).
phi_vals, evals, evecs = ns["fluxonium_phase"](201, EJ, EC, EL, phi_ext)
wavefunction_panel(ax_wf, phi_vals, 0.5*EL*(phi_vals + phi_ext)**2 - EJ*np.cos(phi_vals), evals,
                   [10*evecs[:, i] for i in range(N_STATES)], r"$\phi$", ylim=(-10, 20))
ax_wf.set_xlim(-3*np.pi, np.pi)
ax_wf.set_xticks(np.pi*np.arange(-3, 2))
ax_wf.set_xticklabels([r"$-3\pi$", r"$-2\pi$", r"$-\pi$", r"$0$", r"$\pi$"])

# (b) Lowest four levels vs. external flux (Fock basis).
phi_ext_vals = np.linspace(0, 2*np.pi, 101)
levels = np.array([ns["fluxonium_fock"](80, EJ, EC, EL, pe)[0][:N_STATES] for pe in phi_ext_vals])
dispersion_panel(ax_disp, phi_ext_vals, levels, r"$\phi_{\rm ext}$")
ax_disp.set_xticks([0, np.pi, 2*np.pi])
ax_disp.set_xticklabels([r"$0$", r"$\pi$", r"$2\pi$"])

# (c) Fock basis convergence.
fock_err = ns["fock_err"]
basis_panel(ax_basis, np.array(list(fock_err)), np.array(list(fock_err.values())), "Fock levels")

# (d) Phase grid convergence for different stencil orders.
grid_panel(ax_grid, ns["grid_err"])

out = os.path.join(here, "fluxonium_convergence.pdf")
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
