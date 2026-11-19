#!/usr/bin/env python3
"""
Fig. 1(c): 2D Rashba toy model, velocity-gauge WAVEFUNCTION propagation.

Field along y  ->  A_ext = (0, A(t), 0),  ky -> ky + A(t).
Broken mirror M_y, surviving M_x  ->  S_x develops (axial vector along M_x normal).

We propagate the occupied Bloch orbitals |psi_{n,k}(t)> = U_k(t)|psi_{n,k}(0)>
(time-dependent Schroedinger equation), NOT the density matrix, and sum
S_x(t) = (1/N_k) sum_{k, occ n} (hbar/2) <psi_{n,k}(t)|sigma_x|psi_{n,k}(t)>   [Eq. (7)].

Two SOC variants:
  CN     : SOC uses bare k (spin sector time-independent -> S_x frozen at 0)
  CN+GF  : SOC uses mechanical k+A  (gauge-consistent -> S_x develops)

Conventions (manuscript Methods):
  t=1 eV, a=1 A, alpha=0.3 (= paper's 2*alpha), mu=-1.5 eV (filling 0.245),
  51x51 mesh k = -pi + 2 pi j/51, dt=0.05 hbar/t (~33 as), t_ramp~13.2 fs, t_max~131.6 fs,
  E0 = 0.5 and 1.0 mV/A  ->  eps = eEa/t = 5e-4 and 1e-3.
  A_ext(t) = -c int_0^t E_ext dt'  and  k(t) = k0 + e A_ext(t)/(hbar c)   [Eqs. (10), (6)].
"""

import numpy as np
import os

# --- parameters ---
t_hop  = 1.0
alpha  = 0.3          # = paper's 2*alpha (Rashba coefficient)
mu     = -1.5
Nk     = 51
dt     = 0.05
t_ramp = 20.0         # ~13.2 fs
Nt     = 4000         # t_max = 200 t.u. ~ 131.6 fs
rec    = 10
fs_per_unit = 0.6582 / t_hop

# --- k grid (match rashba_gauge.py) ---
k1  = np.linspace(-np.pi, np.pi, Nk, endpoint=False)
KX, KY = np.meshgrid(k1, k1, indexing='ij')
kx = KX.ravel().astype(np.float64)
ky = KY.ravel().astype(np.float64)
K  = len(kx)
sin_kx = np.sin(kx)          # unaffected by A_y
cos_kx = np.cos(kx)


def A_of_t(t, eps):
    """Vector potential for a smooth turn-on E(t)=eps*[3x^2-2x^3], x=t/t_ramp.
       A(t) = -int_0^t E dt'  (dimensionless, e=hbar=c=a=1)."""
    if t <= 0:
        return 0.0
    if t < t_ramp:
        x = t / t_ramp
        return -eps * t_ramp * (x**3 - x**4 / 2)
    return -eps * t_ramp * 0.5 - eps * (t - t_ramp)


def hfields(A, gf):
    """Return (eps_kin, hx, hy) at shift A. gf=True -> SOC sees k+A (CN+GF)."""
    ky_kin = ky + A
    eps_kin = -2 * t_hop * (cos_kx + np.cos(ky_kin))    # spin-scalar (drops out of spin)
    ky_soc = ky + A if gf else ky
    hx = alpha * np.sin(ky_soc)
    hy = -alpha * sin_kx
    return eps_kin, hx, hy


def init_states():
    """Occupied-band spinors and occupations at t=0 (CN=CN+GF at A=0)."""
    _, hx, hy = hfields(0.0, gf=True)
    h = np.sqrt(hx**2 + hy**2)
    hs = np.maximum(h, 1e-14)
    b_over_h = (hx + 1j*hy) / hs          # lower component of |+>
    inv2 = 1/np.sqrt(2)
    # |+> = (1, b/h)/sqrt2   (E = eps + h),  |-> = (1, -b/h)/sqrt2  (E = eps - h)
    p_up = np.full(K, inv2, dtype=complex); p_dn = inv2 * b_over_h
    m_up = np.full(K, inv2, dtype=complex); m_dn = -inv2 * b_over_h
    eps0, _, _ = hfields(0.0, gf=True)
    Ep, Em = eps0 + h, eps0 - h
    fp = (Ep < mu).astype(float)
    fm = (Em < mu).astype(float)
    print(f"  filling = {(fp.sum()+fm.sum())/(2*K):.3f} "
          f"({fp.sum()+fm.sum():.0f}/{2*K})")
    return (p_up, p_dn, m_up, m_dn, fp, fm)


def U_apply(up, dn, A, gf):
    """Exact 2x2 velocity-gauge propagator over dt, H at midpoint shift A."""
    eps_kin, hx, hy = hfields(A, gf)
    h = np.sqrt(hx**2 + hy**2)
    c = np.cos(h*dt)
    sh = np.where(h > 1e-14, np.sin(h*dt)/np.maximum(h, 1e-14), dt)
    ph = np.exp(-1j*eps_kin*dt)                 # spin-scalar global phase per k
    b = hx + 1j*hy                              # lower-left of (hx sx + hy sy)
    u00 = ph*c
    u11 = ph*c
    u01 = ph*(-1j*sh*(hx - 1j*hy))
    u10 = ph*(-1j*sh*(hx + 1j*hy))
    return u00*up + u01*dn, u10*up + u11*dn


def sx_sum(p_up, p_dn, m_up, m_dn, fp, fm):
    """S_x = (1/K) sum_{occ} (hbar/2) <sigma_x>, Eq. (7):  <sigma_x> = 2 Re(up* dn)."""
    s = fp*2*np.real(np.conj(p_up)*p_dn) + fm*2*np.real(np.conj(m_up)*m_dn)
    return 0.5 * s.sum() / K


def propagate(eps, gf):
    p_up, p_dn, m_up, m_dn, fp, fm = init_states()
    n_rec = Nt // rec + 1
    times = np.zeros(n_rec); Sx = np.zeros(n_rec)
    Sx[0] = sx_sum(p_up, p_dn, m_up, m_dn, fp, fm)
    irec = 0
    for it in range(1, Nt+1):
        t_mid = (it-1)*dt + dt/2
        A = A_of_t(t_mid, eps)
        p_up, p_dn = U_apply(p_up, p_dn, A, gf)
        m_up, m_dn = U_apply(m_up, m_dn, A, gf)
        if it % rec == 0:
            irec += 1
            times[irec] = it*dt
            Sx[irec] = sx_sum(p_up, p_dn, m_up, m_dn, fp, fm)
    return times[:irec+1]*fs_per_unit, Sx[:irec+1]


def main():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(outdir, exist_ok=True)
    print("="*56)
    print("Fig.1(c): VG wavefunction propagation, field||y -> S_x")
    print(f"  Nk={Nk}x{Nk}, alpha={alpha}, mu={mu}, dt={dt}, "
          f"t_max={Nt*dt*fs_per_unit:.1f} fs")
    print("="*56)

    out = {}
    t_ref = None
    for eps, tag in [(5e-4, '0.5mV'), (1e-3, '1.0mV')]:
        print(f"[CN+GF] E0={tag}  (eps={eps:.0e})")
        t, sx = propagate(eps, gf=True)
        out[f'Sx_gf_{tag}'] = sx; t_ref = t
        print(f"   S_x(final) = {sx[-1]:.4e}   S_x(0)={sx[0]:.2e}")
    print("[CN] (field||y, eps=1e-3)")
    t, sx_cn = propagate(1e-3, gf=False)
    out['Sx_cn'] = sx_cn
    print(f"   S_x(final) = {sx_cn[-1]:.4e}   max|S_x| = {np.abs(sx_cn).max():.2e}")

    np.savez(os.path.join(outdir, 'fig1c_velocity_gauge_Nk51.npz'),
             times_fs=t_ref, **out,
             params=np.array([t_hop, alpha, mu, dt, Nk]))
    # plain-text dump
    hdr = "time_fs  Sx_CN  Sx_CNGF_0.5mV  Sx_CNGF_1.0mV   [hbar/cell]"
    dat = np.column_stack([t_ref, out['Sx_cn'],
                           out['Sx_gf_0.5mV'], out['Sx_gf_1.0mV']])
    np.savetxt(os.path.join(outdir, 'fig1c_velocity_gauge_Nk51.dat'), dat, header=hdr)
    print("saved: data/fig1c_velocity_gauge_Nk51.dat (+ .npz)")


if __name__ == '__main__':
    main()
