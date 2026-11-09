"""Eigenstates of the four single-mode qubits of fig:single_mode_qubits, in the phase and charge bases.

Wavefunctions are real: n_g = 0, phi_ext = pi, and fix_phase is applied separately in each basis.

Runs ../../code-snippets/chapter_0.py (the source of the chapter's code listings) for its solvers and
writes single_mode_eigenstates.pdf next to this file.
All four circuits share E_J; the rows share E_C and the columns share E_L, so moving down a column
only adds capacitance and moving across a row only adds the inductor.
"""
import os
import runpy
import numpy as np
from _convergence_plot import *  # sets the Agg backend and the shared colors

here = os.path.dirname(os.path.abspath(__file__))
ns = runpy.run_path(os.path.join(here, "..", "..", "code-snippets", "chapter_0.py"))

EJ, EL = 10, 0.4
EC_SMALL_C, EC_LARGE_C = 5, 0.5  # E_J/E_C = 2 and 20
N_PLOT = 2
N_CUT = 20

plt.rcParams.update({"font.size": 8, "mathtext.fontset": "cm", "font.family": "serif", "axes.linewidth": 0.6})


def real_fixed(psi):
    """Apply fix_phase (first-half maximum real and positive) and check that the result is real."""
    psi = fix_phase(psi)
    assert np.abs(psi.imag).max() < 1e-8*np.abs(psi).max(), "wavefunction is not real after fix_phase"
    return psi.real


def compact_states(EC):
    """Charge-basis eigenstates (n_g = 0) and their phase-basis wavefunctions on [-pi, pi)."""
    n_vals, evals, evecs = ns["transmon_charge"](N_CUT, EJ, EC, 0)
    phi = np.linspace(-np.pi, np.pi, 801)
    psi_phi = np.exp(1j*np.outer(phi, n_vals)) @ evecs[:, :N_PLOT] / np.sqrt(2*np.pi)
    return (evals[:N_PLOT], phi, [real_fixed(psi_phi[:, i]) for i in range(N_PLOT)],
            n_vals, [real_fixed(evecs[:, i].astype(complex)) for i in range(N_PLOT)])


def fluxonium_states(EC):
    """Phase-grid eigenstates at half flux (phi_ext = pi) and their continuous charge wavefunctions.

    The potential 0.5*E_L*(phi + pi)^2 - E_J*cos(phi) is symmetric about phi = -pi, so the Fourier
    transform is taken about that point; the charge wavefunctions are then real up to a global phase.
    """
    phi, evals, evecs = ns["fluxonium_phase"](1201, EJ, EC, EL, np.pi, phi_range=(-6*np.pi, 4*np.pi))
    dphi = phi[1] - phi[0]
    psi = evecs[:, :N_PLOT].T/np.sqrt(dphi)
    n = np.linspace(-3, 3, 601)
    psi_n = psi @ np.exp(-1j*np.outer(phi + np.pi, n))*dphi/np.sqrt(2*np.pi)
    return (evals[:N_PLOT], phi, [real_fixed(p.astype(complex)) for p in psi],
            n, [real_fixed(p) for p in psi_n])


def energy_panel(ax, x, potential, evals, wavefunctions, scale, discrete=False):
    """Potential (dashed) with each real wavefunction drawn at its eigenvalue, as in the convergence figures.

    |0> is drawn thicker so that it stays visible under |1> where the two coincide (heavy fluxonium).
    """
    ax.plot(x, potential, "--", c="gray", alpha=0.5, lw=0.9)
    for i, (E, psi) in enumerate(zip(evals, wavefunctions)):
        c, lw = STATE_COLORS[i], (2.0 if i == 0 else 1.0)
        ax.axhline(E, ls=":", c=c, lw=0.7)
        if discrete:
            ax.vlines(x, E, E + scale*psi, colors=c, lw=lw + 0.6)
        else:
            ax.fill_between(x, E, E + scale*psi, color=c, alpha=0.2, lw=0)
            ax.plot(x, E + scale*psi, c=c, lw=lw)
    ax.set_yticks([])


fig = plt.figure(figsize=(6.5, 3.6))
gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 0.22, 1, 1], hspace=0.55, wspace=0.3)
systems = [  # (row, first column, name, E_C, has inductor)
    (0, 0, "Cooper pair box", EC_SMALL_C, False),
    (1, 0, "Transmon", EC_LARGE_C, False),
    (0, 3, "Light fluxonium", EC_SMALL_C, True),
    (1, 3, "Heavy fluxonium", EC_LARGE_C, True),
]
for row, col, name, EC, inductor in systems:
    ax_phi, ax_n = fig.add_subplot(gs[row, col]), fig.add_subplot(gs[row, col + 1])
    if inductor:
        evals, phi, p_phi, n, p_n = fluxonium_states(EC)
        V_phi = 0.5*EL*(phi + np.pi)**2 - EJ*np.cos(phi)
        ax_phi.set_xlim(-3*np.pi, np.pi)
        ax_phi.set_xticks([-2*np.pi, -np.pi, 0])
        ax_phi.set_xticklabels([r"$-2\pi$", r"$-\pi$", r"$0$"])
        discrete = False
    else:
        evals, phi, p_phi, n, p_n = compact_states(EC)
        V_phi = -EJ*np.cos(phi)
        ax_phi.set_xlim(-np.pi, np.pi)
        ax_phi.set_xticks([-np.pi, 0, np.pi])
        ax_phi.set_xticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
        discrete = True
    # y-range from the bottom of the potential to above |1>; wavefunctions get a common height
    v_min = V_phi[(phi >= ax_phi.get_xlim()[0]) & (phi <= ax_phi.get_xlim()[1])].min()
    height = 0.45*max(evals[1] - evals[0], 0.45*(evals[1] - v_min))  # heavy fluxonium levels nearly degenerate
    lo, hi = min(v_min, evals[0] - 1.1*height), evals[1] + 1.3*height
    lo -= 0.05*(hi - lo)
    energy_panel(ax_phi, phi, V_phi, evals, p_phi, height/np.abs(p_phi).max())
    energy_panel(ax_n, n, 4*EC*n**2, evals, p_n, height/np.abs(p_n).max(), discrete=discrete)
    ax_n.set_xlim(-3, 3)
    ax_n.set_xticks([-3, 0, 3])
    for ax in (ax_phi, ax_n):
        ax.set_ylim(lo, hi)
    ax_phi.set_xlabel(r"phase $\phi$")
    ax_n.set_xlabel(r"charge $n$")
    if col == 0:
        ax_phi.set_ylabel("Energy")
    ax_phi.set_title(name, loc="left", fontsize=9, x=0.0)
    print(f"{name:16s} E_J/E_C={EJ/EC:4.0f} E_L={EL if inductor else 0}: E01={evals[1]-evals[0]:.3g}")

handles = [plt.Line2D([], [], c=STATE_COLORS[i], lw=(2.0 if i == 0 else 1.0), label=rf"$|{i}\rangle$") for i in range(N_PLOT)]
fig.legend(handles=handles, loc="upper center", ncol=N_PLOT, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.02))
out = os.path.join(here, "single_mode_eigenstates.pdf")
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
