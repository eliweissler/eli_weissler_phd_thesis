"""Code snippets for chapter 0 (numerical modeling of superconducting circuits).

Each `# %% name` marker starts a cell (VS Code / Jupytext notebook cell) and is also the range
marker the thesis uses to include that cell as a listing, so the listings are exactly this code.
Run top to bottom with `python chapter_0.py`; requires numpy, scipy and matplotlib.
"""

# %% fluxonium_phase
import numpy as np
import matplotlib.pyplot as plt
def fluxonium_phase(n_pts, EJ, EC, EL, phi_ext, phi_range=(-5*np.pi, 3*np.pi)):
    phi_vals = np.linspace(*phi_range, n_pts)
    dphi = phi_vals[1]-phi_vals[0]
    n2_op = (2*np.eye(n_pts) - np.diag(np.ones(n_pts-1),1) - np.diag(np.ones(n_pts-1),-1))/dphi**2
    H = 4*EC*n2_op + np.diag(0.5*EL*(phi_vals + phi_ext)**2 - EJ*np.cos(phi_vals))
    evals, evecs = np.linalg.eigh(H)
    return phi_vals, evals, evecs
EJ, EC, EL, phi_ext = 10, 0.5, 0.4, np.pi
phi_vals, evals, evecs = fluxonium_phase(201, EJ, EC, EL, phi_ext)
def fix_phase(psi):
    # global phase s.t. the largest point in the first half of the grid is real and positive
    half = psi[:len(psi)//2]
    ref = half[np.argmax(np.abs(half))]
    return psi*np.conj(ref)/np.abs(ref)
# Plotting
plt.figure()
plt.plot(phi_vals, 0.5*EL*(phi_vals + phi_ext)**2 - EJ*np.cos(phi_vals), "--", c="gray", alpha=0.5, label="potential")
n_plot = 4
for i in range(n_plot):
    plt.plot(phi_vals, evals[i] + 10*np.real(fix_phase(evecs[:,i])), label = rf'$|{i}\rangle$')
    plt.axhline(evals[i], ls=":", c=f"C{i}")
plt.legend()
plt.show()

# %% fluxonium_fock
import numpy as np
from scipy.linalg import expm
def fluxonium_fock(n_fock, EJ, EC, EL, phi_ext):
    a = np.diag(np.sqrt(np.arange(1, n_fock)), 1)
    ad = a.T
    phi_zpf = (2*EC/EL)**0.25
    n_zpf = (EL/(32*EC))**0.25
    phi_op = phi_zpf*(ad + a)
    n_op = 1j*n_zpf*(ad - a)
    # cos(phi) from displacement operators D(+-i phi_zpf)
    alpha = 1j*phi_zpf
    D = expm(alpha*ad - np.conj(alpha)*a)
    cos_phi_op = 0.5*(D + D.conj().T)
    shifted = phi_op + phi_ext*np.eye(n_fock)
    H = 4*EC*n_op@n_op + 0.5*EL*shifted@shifted - EJ*cos_phi_op
    return np.linalg.eigh(H)
EJ, EC, EL, phi_ext = 10, 0.5, 0.4, np.pi
evals, evecs = fluxonium_fock(60, EJ, EC, EL, phi_ext)

# %% fluxonium_convergence
def fluxonium_phase_fd(n_pts, EJ, EC, EL, phi_ext, weights, phi_range=(-5*np.pi, 3*np.pi)):
    # as fluxonium_phase, with n^2 from a symmetric stencil weights = (w_0, w_1, ..., w_k)
    phi_vals = np.linspace(*phi_range, n_pts)
    dphi = phi_vals[1]-phi_vals[0]
    d2_op = weights[0]*np.eye(n_pts) + sum(w*(np.eye(n_pts, k=j) + np.eye(n_pts, k=-j)) for j, w in enumerate(weights) if j > 0)
    H = -4*EC*d2_op/dphi**2 + np.diag(0.5*EL*(phi_vals + phi_ext)**2 - EJ*np.cos(phi_vals))
    return phi_vals, np.linalg.eigvalsh(H)
ref = fluxonium_fock(140, EJ, EC, EL, phi_ext)[0][:4]
# Fock basis: error vs. number of levels
fock_err = {n: np.abs(fluxonium_fock(n, EJ, EC, EL, phi_ext)[0][:4] - ref)
            for n in range(10, 101, 5)}
# Phase basis: error vs. grid spacing for the stencils of the table
stencils = {2: (-2, 1),
            4: (-5/2, 4/3, -1/12),
            6: (-49/18, 3/2, -3/20, 1/90),
            8: (-205/72, 8/5, -1/5, 8/315, -1/560)}
grid_err = {}
for order, weights in stencils.items():
    for n_pts in (85, 101, 151, 201, 301, 401, 601, 801):
        phi_vals, evals = fluxonium_phase_fd(n_pts, EJ, EC, EL, phi_ext, weights)
        grid_err[order, phi_vals[1] - phi_vals[0]] = np.abs(evals[:4] - ref).max()

# %% transmon_charge
import numpy as np
def transmon_charge(n_cut, EJ, EC, ng):
    n_vals = np.arange(-n_cut, n_cut+1)
    dim = len(n_vals)
    # cos(phi) = (e^{i phi} + e^{-i phi})/2 shifts n by +-1
    cos_phi_op = 0.5*(np.eye(dim, k=1) + np.eye(dim, k=-1))
    H = np.diag(4*EC*(n_vals - ng)**2) - EJ*cos_phi_op
    return n_vals, *np.linalg.eigh(H)
# lowest three levels vs. offset charge
ng_vals = np.linspace(-0.5, 0.5, 51)
levels = np.array([transmon_charge(10, 5, 0.2, ng)[1][:3] for ng in ng_vals])
# %% end


# %% transmon_phase
import numpy as np
# central first-derivative stencils (w_1, ..., w_k): d/dphi psi_i = sum_j w_j (psi_{i+j} - psi_{i-j}) / dphi
stencils_d1 = {2: (1/2,),
               4: (2/3, -1/12),
               6: (3/4, -3/20, 1/60),
               8: (4/5, -1/5, 4/105, -1/280)}
def transmon_phase_fd(n_pts, EJ, EC, ng, order):
    # periodic grid: phi = -pi and pi are the same point, so leave out the endpoint
    phi_vals = np.linspace(-np.pi, np.pi, n_pts, endpoint=False)
    dphi = phi_vals[1]-phi_vals[0]
    # periodic gauge: psi(phi + 2 pi) = psi(phi), so the stencils wrap around cyclically
    shift = lambda j: np.roll(np.eye(n_pts), j, axis=0)  # (S psi)_i = psi_{i-j}
    w1, w2 = stencils_d1[order], stencils[order]
    d1_op = sum(w*(shift(-j) - shift(j)) for j, w in enumerate(w1, 1))/dphi
    d2_op = (w2[0]*np.eye(n_pts) + sum(w*(shift(j) + shift(-j)) for j, w in enumerate(w2) if j > 0))/dphi**2
    n_op, n2_op = -1j*d1_op, -d2_op  # n = -i d/dphi
    # the offset charge enters only through the Hamiltonian, 4 EC (n - ng)^2 expanded
    H = 4*EC*(n2_op - 2*ng*n_op + ng**2*np.eye(n_pts)) - EJ*np.diag(np.cos(phi_vals))
    return phi_vals, *np.linalg.eigh(H)
# %% end

# %% transmon_convergence
ng = 0.25
ref_tr = transmon_charge(40, 5, 0.2, ng)[1][:3]
# charge basis: error vs. charge cutoff
charge_err = {n: np.abs(transmon_charge(n, 5, 0.2, ng)[1][:3] - ref_tr)
              for n in range(2, 16)}
# phase basis (periodic grid): error vs. grid spacing for the stencils of the table
grid_err_tr = {}
for order in stencils:
    for n_pts in (21, 31, 42, 52, 63, 84, 105, 140, 209):
        phi_vals, evals, _ = transmon_phase_fd(n_pts, 5, 0.2, ng, order)
        grid_err_tr[order, phi_vals[1] - phi_vals[0]] = np.abs(evals[:3] - ref_tr).max()
# %% end

# %% coupled_capacitive
import numpy as np
def coupled_levels(E1, V1, O1, E2, V2, O2, g, N1, N2, n_levels=10):
    # hierarchical diagonalization: keep the lowest N1 (N2) eigenstates of each subsystem
    # and express the coupling g O1 (x) O2 in those eigenbases
    V1, V2 = V1[:, :N1], V2[:, :N2]
    H = np.kron(np.diag(E1[:N1]), np.eye(N2)) + np.kron(np.eye(N1), np.diag(E2[:N2]))
    H = H + g*np.kron(V1.conj().T @ O1 @ V1, V2.conj().T @ O2 @ V2)
    return np.linalg.eigvalsh(H)[:n_levels]
# subsystems: fluxonium (Fock basis) and transmon (charge basis), diagonalized separately
n_fock, n_cut = 60, 10
E_f, V_f = fluxonium_fock(n_fock, EJ, EC, EL, phi_ext)
n_vals, E_t, V_t = transmon_charge(n_cut, 5, 0.2, 0)
# charge operators in the bare bases
a = np.diag(np.sqrt(np.arange(1, n_fock)), 1)
n_f = 1j*(EL/(32*EC))**0.25*(a.T - a)
n_t = np.diag(n_vals)
# capacitive coupling H_int = g n_f n_t
g = 0.1
couple = lambda N1, N2: coupled_levels(E_f, V_f, n_f, E_t, V_t, n_t, g, N1, N2)
ref_cap = couple(n_fock, 2*n_cut + 1)  # no truncation of the subsystem bases
N_f_vals, N_t_vals = range(4, 31), range(4, 22)  # N_f N_t >= n_levels
err_cap = np.array([[np.abs(couple(N_f, N_t) - ref_cap).max() for N_t in N_t_vals]
                    for N_f in N_f_vals])
# %% end
