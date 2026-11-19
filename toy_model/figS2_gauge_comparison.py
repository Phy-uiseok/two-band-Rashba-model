#!/usr/bin/env python3
"""
Supplementary Fig. S2: gauge comparison for the two-band Rashba model.

Model (spin basis, one site per cell, so the basis Berry connection vanishes):

    H(k) = -2t [cos(kx) + cos(ky)] s0 + alpha [ sin(ky) sx - sin(kx) sy ]

with alpha = 2*alpha_manuscript, i.e. the code's alpha = 0.3 eV is the
manuscript's 2*alpha = 0.3 eV (Rashba coefficient alpha_R = 2*alpha*a = 0.3 eV A).

The field is applied along y (Methods), so the Edelstein response appears in S_x.
Three propagations of the one-particle density matrix rho(k,t) (2x2 at each k):

  VG-full (CN+GF) : velocity gauge, A(t) enters BOTH the hopping and the Rashba
                    term, ky -> ky + A(t)                       [gauge consistent]
  VG-CN           : velocity gauge, A(t) enters the hopping only; the Rashba term
                    keeps the bare k                            [canonical only]
  DG              : dipole (length) gauge, bare-k Hamiltonian plus e E(t) . r with
                    r = i d/dk, propagated by Strang splitting
                    rho <- U T_delta U rho U^+ T_delta^+ U^+,
                    U = exp[-i h(k).s dt/2], T_delta an exact spectral shift of ky

The spin-scalar part eps(k) commutes with rho and drops out of all three.

Units (declared convention): t = 1 eV, a = 1 A, hbar = e = 1.
  1 time unit = hbar/t = 0.65821 fs ; eps = e E a / t, so eps = 1e-3 <-> E0 = 1 mV/A
  S [hbar/cell] = (1/2) Tr[s rho] / N_k

Usage:  python3 figS2_gauge_comparison.py [Nk]
"""

import os
import sys

import numpy as np

# --- parameters (manuscript Methods) ---
t_hop  = 1.0
alpha  = 0.3           # = manuscript 2*alpha
mu     = -1.5
Nk     = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 51
dt     = 0.05          # ~33 as
t_ramp = 20.0          # ~13.2 fs
Nt     = 4000          # t_max = 200 t.u. ~ 131.6 fs
rec    = 10
eps0   = 1e-3          # E0 = 1.0 mV/A
FS_PER_UNIT = 0.6582119569 / t_hop

# --- 51 x 51 k mesh, k = -pi + 2 pi j/Nk (j = 0 .. Nk-1): no duplicated zone-boundary point ---
k1 = np.linspace(-np.pi, np.pi, Nk, endpoint=False)
KX, KY = np.meshgrid(k1, k1, indexing='ij')
NKTOT = KX.size
DRIVE_AXIS = 1                                    # field along y -> shift ky
dk = 2 * np.pi / Nk
QDRIVE = 2 * np.pi * np.fft.fftfreq(Nk, d=dk)     # conjugate index along the driven axis


def A_of_t(t, eps):
    """A(t) = -int_0^t E dt' for the smooth turn-on E(t) = eps [3x^2 - 2x^3], x = t/t_ramp."""
    if t <= 0:
        return 0.0
    if t < t_ramp:
        x = t / t_ramp
        return -eps * t_ramp * (x**3 - x**4 / 2)
    return -eps * t_ramp * 0.5 - eps * (t - t_ramp)


def E_of_t(t, eps):
    if t <= 0:
        return 0.0
    if t < t_ramp:
        x = t / t_ramp
        return eps * (3 * x**2 - 2 * x**3)
    return eps


def h_of_k(kx, ky):
    """(hx, hy) of the Rashba field and the spin-scalar eps(k)."""
    return alpha * np.sin(ky), -alpha * np.sin(kx)


def ground_state():
    """rho(k, 0) = f_+ P_+ + f_- P_-, P_pm = (1/2)(I pm n.s), n = h/|h|."""
    hx, hy = h_of_k(KX, KY)
    h = np.sqrt(hx**2 + hy**2)
    hs = np.maximum(h, 1e-14)
    nx = np.where(h > 1e-12, hx / hs, 1.0)        # n is arbitrary where h = 0 (there f_+ = f_-)
    ny = np.where(h > 1e-12, hy / hs, 0.0)
    ekin = -2 * t_hop * (np.cos(KX) + np.cos(KY))
    fp = (ekin + h < mu).astype(float)
    fm = (ekin - h < mu).astype(float)
    rho = np.zeros(KX.shape + (2, 2), dtype=complex)
    s = 0.5 * (fp + fm)                           # (f_+ + f_-)/2 * I
    d = 0.5 * (fp - fm)                           # (f_+ - f_-)/2 * n.s
    rho[..., 0, 0] = s + 0.0
    rho[..., 1, 1] = s - 0.0
    rho[..., 0, 1] = d * (nx - 1j * ny)
    rho[..., 1, 0] = d * (nx + 1j * ny)
    print("  filling = %.3f (%d/%d)" % ((fp.sum() + fm.sum()) / (2 * NKTOT),
                                        fp.sum() + fm.sum(), 2 * NKTOT))
    return rho


def rotate(rho, hx, hy, tau):
    """rho <- U rho U^+ with U = exp[-i (hx sx + hy sy) tau] (closed form, exact)."""
    h = np.sqrt(hx**2 + hy**2)
    hs = np.maximum(h, 1e-300)
    nx, ny = hx / hs, hy / hs
    c = np.cos(h * tau)
    s = np.sin(h * tau)
    # U = c I - i s (n.s);  n.s = [[0, nx - i ny], [nx + i ny, 0]]
    u00 = c.astype(complex)
    u11 = c.astype(complex)
    u01 = -1j * s * (nx - 1j * ny)
    u10 = -1j * s * (nx + 1j * ny)
    U = np.stack([np.stack([u00, u01], axis=-1),
                  np.stack([u10, u11], axis=-1)], axis=-2)
    return U @ rho @ np.conj(np.swapaxes(U, -1, -2))


def shift_drive(rho, delta):
    """exact spectral translation of rho along the driven k axis"""
    F = np.fft.fft(rho, axis=DRIVE_AXIS)
    ph = np.exp(1j * QDRIVE * delta)[None, :, None, None]
    return np.fft.ifft(F * ph, axis=DRIVE_AXIS)


def spin(rho):
    """S = (1/2) Tr[s rho] / N_k   [hbar/cell]"""
    r01, r10 = rho[..., 0, 1], rho[..., 1, 0]
    r00, r11 = rho[..., 0, 0], rho[..., 1, 1]
    return 0.5 * np.array([np.real(r01 + r10).sum(),
                           np.real(1j * (r01 - r10)).sum(),
                           np.real(r00 - r11).sum()]) / NKTOT


def propagate(mode, eps=eps0, n_steps=Nt):
    rho = ground_state()
    n_rec = n_steps // rec + 1
    times = np.zeros(n_rec)
    S = np.zeros((n_rec, 3))
    S[0] = spin(rho)
    irec = 0
    hx0, hy0 = h_of_k(KX, KY)
    for it in range(1, n_steps + 1):
        t_mid = (it - 1) * dt + dt / 2
        if mode == 'VGfull':
            hx, hy = h_of_k(KX, KY + A_of_t(t_mid, eps))
            rho = rotate(rho, hx, hy, dt)
        elif mode == 'VGCN':
            rho = rotate(rho, hx0, hy0, dt)        # A enters only the s0 hopping -> no spin dynamics
        elif mode == 'DG':
            delta = E_of_t(t_mid, eps) * dt
            rho = rotate(rho, hx0, hy0, dt / 2)
            rho = shift_drive(rho, delta)
            rho = rotate(rho, hx0, hy0, dt / 2)
        else:
            raise ValueError(mode)
        if it % rec == 0:
            irec += 1
            times[irec] = it * dt
            S[irec] = spin(rho)
    return times[:irec + 1] * FS_PER_UNIT, S[:irec + 1]


def main():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(outdir, exist_ok=True)
    print("=" * 62)
    print("Fig. S2: gauge comparison, field || y -> S_x")
    print("  Nk = %dx%d, alpha = %g, mu = %g, dt = %g, t_max = %.1f fs, E0 = %.1f mV/A"
          % (Nk, Nk, alpha, mu, dt, Nt * dt * FS_PER_UNIT, eps0 * 1e3))
    print("=" * 62)

    out = {}
    tref = None
    for mode in ['DG', 'VGfull', 'VGCN']:
        print("[%s]" % mode)
        t, S = propagate(mode)
        tref = t
        out[mode] = S
        print("   S_x(final) = %+.6e   max|S_y| = %.2e   max|S_z| = %.2e"
              % (S[-1, 0], np.abs(S[:, 1]).max(), np.abs(S[:, 2]).max()))

    ref = np.abs(out['VGfull'][:, 0]).max()
    dev = np.abs(out['DG'][:, 0] - out['VGfull'][:, 0]).max()
    print("\n  DG vs VG-full : max deviation = %.3e hbar/cell = %.3f %% of the peak"
          % (dev, 100 * dev / ref))
    print("  VG-CN         : max|S_x| = %.3e hbar/cell = %.3f %% of the peak"
          % (np.abs(out['VGCN'][:, 0]).max(),
             100 * np.abs(out['VGCN'][:, 0]).max() / ref))

    hdr = ("Fig. S2  two-band Rashba model, field || y, E0 = %.1f mV/A, %dx%d mesh\n"
           "t = 1 eV, a = 1 A, alpha(code) = 2*alpha(manuscript) = %g eV, mu = %g eV, dt = %g hbar/t\n"
           "time_fs   Sx_DG   Sx_VGfull   Sx_VGCN     [hbar/cell]" % (eps0 * 1e3, Nk, Nk, alpha, mu, dt))
    dat = np.column_stack([tref, out['DG'][:, 0], out['VGfull'][:, 0], out['VGCN'][:, 0]])
    np.savetxt(os.path.join(outdir, 'figS2_gauge_comparison_Nk%d.dat' % Nk), dat, header=hdr)
    np.savez(os.path.join(outdir, 'figS2_gauge_comparison_Nk%d.npz' % Nk),
             times_fs=tref, S_DG=out['DG'], S_VGfull=out['VGfull'], S_VGCN=out['VGCN'],
             params=np.array([t_hop, alpha, mu, dt, Nk, eps0]))
    print("saved: %s/figS2_gauge_comparison_Nk%d.dat (+ .npz)" % (outdir, Nk))


if __name__ == '__main__':
    main()
