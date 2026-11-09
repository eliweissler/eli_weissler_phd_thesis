"""Shared plotting helpers for the four-panel numerical-modeling figures of chapter 0."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

N_STATES = 4
STATE_COLORS = [f"C{i}" for i in range(N_STATES)]


def new_figure():
    """2x2 figure: (a) wavefunctions, (b) dispersion, (c) basis convergence, (d) grid convergence."""
    plt.rcParams.update({"font.size": 8, "mathtext.fontset": "cm", "font.family": "serif", "axes.linewidth": 0.6})
    fig = plt.figure(figsize=(6.5, 5.4))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.34)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    for ax, tag in zip(axes, "abcd"):
        ax.set_title(f"({tag})", loc="left", fontsize=9)
    return fig, axes


def fix_phase(psi):
    """Global phase such that the largest-magnitude point in the first half of the grid is positive and real."""
    half = psi[:len(psi)//2]
    ref = half[np.argmax(np.abs(half))]
    return psi*np.conj(ref)/np.abs(ref)


def wavefunction_panel(ax, x, potential, evals, wavefunctions, xlabel, ylim, legend_kw=None):
    """Potential, eigenvalues (dotted) and wavefunctions offset by their eigenvalues.

    Each (scaled) wavefunction is plotted with its phase fixed by `fix_phase`.
    """
    ax.plot(x, potential, "--", c="gray", alpha=0.5, label="potential")
    for i in range(len(wavefunctions)):
        ax.plot(x, evals[i] + np.real(fix_phase(wavefunctions[i])), c=STATE_COLORS[i], label=rf"$|{i}\rangle$")
        ax.axhline(evals[i], ls=":", c=STATE_COLORS[i], lw=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Energy")
    ax.set_ylim(*ylim)
    ax.legend(**{"ncol": 3, "fontsize": 6.5, "frameon": False, "loc": "upper center", **(legend_kw or {})})


def dispersion_panel(ax, x, levels, xlabel):
    for i in range(levels.shape[1]):
        ax.plot(x, levels[:, i], c=STATE_COLORS[i], lw=1.2, label=rf"$|{i}\rangle$")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Energy")


def basis_panel(ax, sizes, err, xlabel):
    """Per-state error of the lowest eigenvalues vs. basis size."""
    for i in range(err.shape[1]):
        ax.semilogy(sizes, np.maximum(err[:, i], 1e-16), "o-", ms=3, lw=1, c=STATE_COLORS[i], label=rf"$|{i}\rangle$")
    ax.set_xlabel(xlabel)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_ylabel(r"$|E_i - E_i^{\rm ref}|$")
    ax.legend(frameon=False, fontsize=7)


def grid_panel(ax, grid_err):
    """Largest error vs. grid spacing, one curve per stencil order, over exactly one decade in dphi."""
    orders = sorted({o for o, _ in grid_err})
    cmap = plt.cm.viridis
    for k, order in enumerate(orders):
        dphi = np.array([d for (o, d) in grid_err if o == order])
        err = np.array([e for (o, _), e in grid_err.items() if o == order])
        color = cmap(0.1 + 0.75*k/(len(orders) - 1))
        ax.loglog(dphi, np.maximum(err, 1e-16), "o-", ms=3, lw=1, c=color, label=rf"order {order}")
        ax.loglog(dphi, err[0]*(dphi/dphi[0])**order, ":", lw=0.8, c=color)  # slope-p guide
        slopes = np.diff(np.log(err[:4]))/np.diff(np.log(dphi[:4]))
        print(f"order {order}: slopes {np.round(slopes, 2)}, last err {err[-1]:.1e}")
    ax.set_ylim(1e-13, 1)
    ax.set_xlim(0.03, 0.3)  # one decade in dphi, so slope p = p decades of error per decade
    ax.set_xticks([0.03, 0.1, 0.3])
    ax.set_xticklabels(["0.03", "0.1", "0.3"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10, subs=[1.0], numticks=30))
    ax.grid(which="both", axis="y", lw=0.4, alpha=0.4)
    ax.set_xlabel(r"$\Delta\phi$")
    ax.set_ylabel(r"$\max_i |E_i - E_i^{\rm ref}|$")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
