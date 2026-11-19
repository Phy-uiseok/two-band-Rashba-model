# Essential Role of Gauge-Consistent Spin–Orbit Coupling in Field-Driven Symmetry Breaking and Spin Dynamics

Code and data for the manuscript *Essential Role of Gauge-Consistent Spin–Orbit Coupling in Field-Driven
Symmetry Breaking and Spin Dynamics* (npj Computational Materials, under review).

| directory | contents |
|---|---|
| `toy_model/` | two self-contained Python scripts that reproduce Fig. 1c and Supplementary Fig. S2 with the parameters given in the Methods section |

Only `numpy` is required for the Python scripts; each run takes about a minute on one core.

## Two-band Rashba model

```
H(k) = -2t [cos(kx) + cos(ky)] s0 + 2a [ sin(ky) sx - sin(kx) sy ]
```

| parameter | value | | parameter | value |
|---|---|---|---|---|
| `t` | 1 eV | | mesh | 51 x 51 |
| `2a` | 0.3 eV (`alpha` in the scripts) | | `dt` | 0.05 hbar/t = 33 as |
| `mu` | -1.5 eV (filling 0.245) | | `t_max` | 131.6 fs |
| field | along `y`, smooth turn-on, `tau` = 13.2 fs | | `E0` | 0.5 and 1.0 mV/A |

The mesh is generated with `np.linspace(-pi, pi, Nk, endpoint=False)`, so the two zone
boundaries are not counted twice. The field enters through
`A(t) = -c int_0^t E dt'` and `k(t) = k0 + e A(t)/(hbar c)`; with this convention a field
along `+y` gives a positive `S_x`. Spin is reported as `S = (1/N_k) sum_occ (hbar/2) <sigma>`
in units of hbar per cell.

```
python3 toy_model/fig1c_velocity_gauge.py      # Fig. 1c: VG wavefunction propagation, CN vs CN+GF
python3 toy_model/figS2_gauge_comparison.py    # Fig. S2: dipole gauge vs velocity gauge, density matrix
```

The wavefunction curves of Fig. S2 are the 1.0 mV/A curves of `fig1c_velocity_gauge.py`; the
density-matrix curves are those of `figS2_gauge_comparison.py`.

Values at `t` = 131.6 fs for `E0` = 1.0 mV/A, 51 x 51 (units of hbar per cell):

| propagation | SOC Hamiltonian | `S_x` |
|---|---|---|
| velocity gauge, wavefunction | canonical only | 6e-18 (zero) |
| velocity gauge, wavefunction | CN + GF | +2.1924e-3 |
| velocity gauge, density matrix | CN + GF | +2.1924e-3 |
| dipole (length) gauge, density matrix | canonical | +2.1837e-3 |

The dipole gauge and the gauge-consistent velocity gauge agree to 0.73 % of the peak on
this mesh (1.6e-5 hbar per cell) and converge to each other as the mesh is refined, whereas
the truncated velocity gauge gives exactly zero: in this two-band model the entire Edelstein
response is carried by the gauge-field part of the Rashba term.

## License

The Python scripts and the data under `toy_model/data/` are released under the MIT license and
CC-BY-4.0, respectively.
