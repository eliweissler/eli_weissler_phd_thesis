"""Convergence of hierarchical diagonalization for the capacitively coupled fluxonium and transmon.

Runs ../../code-snippets/chapter_0.py (the source of the chapter's code listings) and plots
(a) `err_cap`, the error vs. the retained fluxonium and transmon states, and
(b) the error with a fixed number of retained states vs. the charging energies of both subsystems, which
repeats the calculation of the hierarchical-diagonalization listing on a grid of charging energies.
Writes hierarchical_convergence.pdf next to this file (takes about a minute).
"""
import os
import runpy
import numpy as np
import matplotlib.colors
from _convergence_plot import plt

here = os.path.dirname(os.path.abspath(__file__))
ns = runpy.run_path(os.path.join(here, "..", "..", "code-snippets", "chapter_0.py"))

# (a) err[i, j]: error with N_f_vals[i] fluxonium and N_t_vals[j] transmon states kept
err_cap = ns["err_cap"]
N_f_vals, N_t_vals = list(ns["N_f_vals"]), list(ns["N_t_vals"])

# (b) fixed truncation, scan of the charging energies (both subsystems are re-diagonalized at every point)
fluxonium_fock, transmon_charge, coupled_levels = ns["fluxonium_fock"], ns["transmon_charge"], ns["coupled_levels"]
EJ, EC, EL, phi_ext, g = ns["EJ"], ns["EC"], ns["EL"], ns["phi_ext"], ns["g"]
n_fock, n_cut = ns["n_fock"], ns["n_cut"]
EJ_t, EC_t0 = 5, 0.2  # transmon parameters of the listings
N_f, N_t = 8, 5  # retained fluxonium and transmon states

a = np.diag(np.sqrt(np.arange(1, n_fock)), 1)
EC_f_vals = np.linspace(0.2, 1.6, 9)
EC_t_vals = np.linspace(0.1, 0.5, 9)
err_ec = np.zeros((len(EC_f_vals), len(EC_t_vals)))
for i, EC_f in enumerate(EC_f_vals):
    E_f, V_f = fluxonium_fock(n_fock, EJ, EC_f, EL, phi_ext)
    n_f = 1j*(EL/(32*EC_f))**0.25*(a.T - a)
    for j, EC_t in enumerate(EC_t_vals):
        n_vals, E_t, V_t = transmon_charge(n_cut, EJ_t, EC_t, 0)
        args = (E_f, V_f, n_f, E_t, V_t, np.diag(n_vals), g)
        ref = coupled_levels(*args, n_fock, 2*n_cut + 1)  # no truncation of the subsystem bases
        err_ec[i, j] = np.abs(coupled_levels(*args, N_f, N_t) - ref).max()
print("error range (b)", err_ec.min(), err_ec.max())


def heatmap(ax, err, x_vals, y_vals, norm, xlabel, ylabel):
    dx, dy = x_vals[1] - x_vals[0], y_vals[1] - y_vals[0]
    extent = (x_vals[0] - dx/2, x_vals[-1] + dx/2, y_vals[0] - dy/2, y_vals[-1] + dy/2)
    im = ax.imshow(err, origin="lower", extent=extent, aspect="auto", cmap="viridis_r", norm=norm)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label(r"$\max_i |E_i - E_i^{\rm ref}|$")
    return cbar


plt.rcParams.update({"font.size": 8, "mathtext.fontset": "cm", "font.family": "serif", "axes.linewidth": 0.6})
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.5, 2.9), gridspec_kw={"wspace": 0.45})

heatmap(ax_a, err_cap, np.array(N_t_vals), np.array(N_f_vals), matplotlib.colors.LogNorm(vmin=err_cap.min(), vmax=err_cap.max()),
        r"retained transmon states $N_t$", r"retained fluxonium states $N_f$")
ax_a.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax_a.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

cbar_b = heatmap(ax_b, err_ec, EC_t_vals, EC_f_vals, matplotlib.colors.LogNorm(vmin=err_ec.min(), vmax=err_ec.max()),
                 r"transmon $E_C$", r"fluxonium $E_C$")
lo, hi = err_ec.min(), err_ec.max()
ticks = [m*10.0**k for k in range(-12, 0) for m in (1, 2, 5) if lo <= m*10.0**k <= hi]
cbar_b.set_ticks(ticks)
cbar_b.set_ticklabels([rf"$10^{{{k}}}$" if m == 1 else rf"${m}\times10^{{{k}}}$"
                       for t in ticks for k in [int(np.floor(np.log10(t) + 1e-9))] for m in [round(t/10.0**k)]])
cbar_b.ax.minorticks_off()
ax_b.plot(EC_t0, EC, "*", c="white", mec="black", ms=8)  # parameters used in the listings

for ax, tag in ((ax_a, "a"), (ax_b, "b")):
    ax.set_title(f"({tag})", loc="left", fontsize=9)

out = os.path.join(here, "hierarchical_convergence.pdf")
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
