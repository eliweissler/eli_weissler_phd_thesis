"""Matrix elements and sweet-spot curvatures of single-mode qubits on the axes of fig:single_mode_qubits.

Energies are in units of E_C (fixed), with axes E_J/E_C and E_L/E_C. The compact circuits (E_L = 0, Cooper pair box to transmon) form a strip
at n_g = 0; the fluxonium map is at half flux, phi_ext = pi. Writes one square-map panel per quantity:
single_mode_n01.pdf, single_mode_phi01.pdf and single_mode_curvature.pdf.

Off-diagonal: |<0|n|1>| for both, |<0|phi|1>| for fluxonium only.
Diagonal: |d^2 E_01 / d lambda^2| at the sweet spot, lambda = n_g (strip) or phi_ext (map), from a
central finite difference with a per-point step (fd_step), checked against perturbation theory.
"""
import os
import numpy as np
from scipy.linalg import eigh, eigh_tridiagonal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.colors import LogNorm

here = os.path.dirname(os.path.abspath(__file__))
RATIOS = np.geomspace(0.5, 50, 41)        # E_J/E_C
EL_RATIOS = np.geomspace(0.01, 3, 31)     # E_L/E_C
N_CUT = 30
N_STATES = 20                             # states kept in the perturbative cross-check
def edges(c):
    """Cell edges for log-spaced centers."""
    m = np.sqrt(c[1:]*c[:-1])
    return np.concatenate([[c[0]**2/m[0]], m, [c[-1]**2/m[-1]]])



def fd_step(E, dH):
    """Finite-difference step: 1% of the smallest gap among the lowest three levels over their largest
    coupling |<i|dH/dlambda|j>|, capped at 1e-2. For the heavy fluxonium this is E_01 over the
    persistent current E_L|<0|phi|1>|, the width of the parabolic region of E_01 around the sweet spot."""
    gap = min(E[1] - E[0], E[2] - E[1])
    coupling = max(abs(dH[0, 1]), abs(dH[1, 2]), abs(dH[0, 2]))
    return min(1e-2, 0.01*gap/coupling)


def curvature_pt(E, dH):
    """d^2 E_01/dlambda^2 from second-order perturbation theory (the d^2H/dlambda^2 term cancels)."""
    d2 = []
    for kk in (0, 1):
        m = np.arange(len(E)) != kk
        d2.append(2*np.sum(dH[m, kk]**2/(E[kk] - E[m])))
    return abs(d2[1] - d2[0])


def compact(ratio):
    """|<0|n|1>| and |d^2 E_01/dn_g^2| at n_g = 0: (n01, finite difference, perturbation theory).

    Solved with E_J = 1; curvatures are returned in units of E_C."""
    EC = 1/ratio
    n = np.arange(-N_CUT, N_CUT + 1)
    hop = -0.5*(np.eye(len(n), k=1) + np.eye(len(n), k=-1))
    H = lambda ng: np.diag(4*EC*(n - ng)**2) + hop
    E, V = eigh(H(0.0))
    dH = V.T @ ((-8*EC*n)[:, None]*V)  # dH/dn_g in the eigenbasis
    n01 = abs(dH[0, 1])/(8*EC)
    e01 = lambda ng: np.diff(eigh(H(ng), eigvals_only=True, subset_by_index=[0, 1]))[0]
    h = fd_step(E, dH)
    curv_fd = abs(e01(h) + e01(-h) - 2*(E[1] - E[0]))/h**2
    return n01, ratio*curv_fd, ratio*curvature_pt(E, dH)


def fluxonium(EL_EC, ratio):
    """|<0|n|1>|, |<0|phi|1>| and |d^2 E_01/dphi_ext^2| at phi_ext = pi on a 3-point phase grid:
    (n01, phi01, finite difference, perturbation theory).

    With u = phi + pi the potential at phi_ext = pi + d is E_L (u + d)^2/2 + cos(u), and
    dH/dphi_ext = E_L u at the sweet spot. Solved with E_J = 1 (E_L = E_L/E_C over E_J/E_C);
    curvatures are returned in units of E_C.
    """
    EC, EL = 1/ratio, EL_EC/ratio
    U = np.sqrt(2*(2 + 6*np.sqrt(8*EC))/EL) + 4
    dx = min(0.05, 0.08*(8*EC)**0.25)
    u = np.arange(-U, U + dx/2, dx)
    off = np.full(len(u) - 1, -4*EC/dx**2)
    diag = lambda d: 8*EC/dx**2 + 0.5*EL*(u + d)**2 + np.cos(u)
    E, v = eigh_tridiagonal(diag(0.0), off, select="i", select_range=(0, N_STATES - 1))
    n01 = abs(v[:, 0] @ np.gradient(v[:, 1], dx))
    U_mat = v.T @ (u[:, None]*v)
    dH = EL*U_mat
    e01 = lambda d: np.diff(eigh_tridiagonal(diag(d), off, eigvals_only=True, select="i", select_range=(0, 1)))[0]
    h = fd_step(E, dH)
    curv_fd = abs(e01(h) + e01(-h) - 2*(E[1] - E[0]))/h**2
    return n01, abs(U_mat[0, 1]), ratio*curv_fd, ratio*curvature_pt(E, dH)


if __name__ == "__main__":
    strip = np.array([compact(r) for r in RATIOS])                       # (ratio, [n01, curv_fd, curv_pt])
    fmap = np.array([[fluxonium(x, r) for x in EL_RATIOS] for r in RATIOS])  # (ratio, EL, [n01, phi01, fd, pt])
    for name, fd, pt in (("charge", strip[:, 1], strip[:, 2]), ("flux", fmap[:, :, 2], fmap[:, :, 3])):
        print(f"{name} curvature: max |FD/PT - 1| = {np.max(np.abs(fd/pt - 1)):.2e}")
    strip = strip[:, :2]
    fmap = fmap[:, :, :3]

    plt.rcParams.update({"font.size": 8, "mathtext.fontset": "cm", "font.family": "serif", "axes.linewidth": 0.6})
    panels = [  # (file name, colorbar label, strip data or None, map data, log color scale)
        ("single_mode_n01.pdf", r"$|\langle 0|\hat n|1\rangle|$", strip[:, 0], fmap[:, :, 0], True),
        ("single_mode_phi01.pdf", r"$|\langle 0|\hat\phi|1\rangle|$", None, fmap[:, :, 1], False),
        ("single_mode_curvature.pdf", r"$|\partial^2_\lambda E_{01}|/E_C$", strip[:, 1], fmap[:, :, 2], True),
    ]
    ytick = [0.5, 1, 2, 5, 10, 20, 50]
    for fname, title, sdata, mdata, log in panels:
        # one panel per file, all with the same width so the maps share a scale:
        # E_L = 0 strip | break gap | square E_L/E_C map | gap | colorbar
        fig = plt.figure(figsize=(2.75, 2.2))
        gs = fig.add_gridspec(1, 5, wspace=0.0, width_ratios=[0.13, 0.06, 1, 0.07, 0.06])
        vals = mdata.ravel() if sdata is None else np.concatenate([mdata.ravel(), sdata])
        norm = LogNorm(vmin=vals[vals > 0].min(), vmax=vals.max()) if log else plt.Normalize(0, vals.max())
        ax_map = fig.add_subplot(gs[0, 2])
        mesh = ax_map.pcolormesh(edges(EL_RATIOS), edges(RATIOS), np.maximum(mdata, norm.vmin), norm=norm,
                                 cmap="viridis", shading="flat", rasterized=True)
        ax_map.set_xscale("log")
        ax_map.set_yscale("log")
        ax_map.set_ylim(edges(RATIOS)[-1], edges(RATIOS)[0])  # E_J/E_C increases downward, as in the landscape
        ax_map.set_yticks(ytick)
        ax_map.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax_map.set_xlabel(r"$E_L/E_C$", labelpad=1)
        ax_s = fig.add_subplot(gs[0, 0], sharey=ax_map)
        ax_s.set_xlim(0, 1)
        if sdata is not None:
            plt.setp(ax_map.get_yticklabels(), visible=False)
            ax_s.pcolormesh([0, 1], edges(RATIOS), np.maximum(sdata, norm.vmin)[:, None], norm=norm,
                            cmap="viridis", shading="flat", rasterized=True)
            ax_s.set_xticks([0.5])
            ax_s.set_xticklabels([r"$0$"])
            # axis break between E_L = 0 and the fluxonium map
            for ax, side in ((ax_s, 1), (ax_map, 0)):
                ax.plot([side - 0.04, side + 0.04], [-0.03, 0.03], transform=ax.transAxes, c="k", lw=0.6,
                        clip_on=False)
            ax_s.set_yticklabels([f"{t:g}" for t in ytick])
            ax_s.set_ylabel(r"$E_J/E_C$")
        else:
            ax_s.axis("off")
            ax_map.set_yticklabels([f"{t:g}" for t in ytick])
            ax_map.set_ylabel(r"$E_J/E_C$")
        cax = fig.add_subplot(gs[0, 4])
        cb = fig.colorbar(mesh, cax=cax, orientation="vertical")
        cb.set_label(title, fontsize=8)
        cb.ax.tick_params(labelsize=6.5)
        # make the map square: adjust the figure height until its height matches its width
        for _ in range(5):
            pos = ax_map.get_position()
            w_in, h_in = pos.width*fig.get_figwidth(), pos.height*fig.get_figheight()
            fig.set_figheight(fig.get_figheight() + (w_in - h_in))
        out = os.path.join(here, fname)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print("wrote", out)
