import numpy as np
import matplotlib.pyplot as plt

# 물리 상수 및 파라미터, in atomic Rydberg units
e = np.sqrt(2)
ry2ev = 13.6
ry2fs = 4.84e-2
ry2VpA = 36.36
hbar = 1.0
m = 0.5
fine_const = 137.035999084
c = 2 / fine_const
alat = 1.0
t = 1.0 / ry2ev # hopping parameter
alpha = 0.5 / ry2ev
timestep = 10000
dt = 0.50

def rashba_tb_hamiltonian(kx, ky, lam, t=1.0):

    chem_pot = 2.0 / ry2ev  # 화학 퍼텐셜, 필요시 조정 가능
    
    # Identity and Pauli matrices
    I = np.eye(2)
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])

    # Kinetic term
    eps_k = (-2 * t * (np.cos(kx) + np.cos(ky)) + chem_pot) * I

    # Rashba SOC term
    rashba_term = 2 * lam * (np.sin(ky) * sigma_x - np.sin(kx) * sigma_y)

    H = eps_k + rashba_term
    return H

# k-space mesh
nkx = 100
nky = 100
kx = np.linspace(-1*np.pi, 1*np.pi, nkx)
ky = np.linspace(-1*np.pi, 1*np.pi, nky)
KX, KY = np.meshgrid(kx, ky)

# 고유값 저장 배열 (에너지 밴드)
E = np.zeros((nky, nkx, 2))

# (옵션) 고유벡터 저장
eigvecs = np.zeros((nky, nkx, 2, 2), dtype=complex)
# occupation 수 저장용 배열
occ = np.zeros((nky, nkx, 2))

# 루프 수행
for ix in range(nkx):
    for iy in range(nky):
        kxi = KX[ix, iy]
        kyi = KY[ix, iy]
        H = rashba_tb_hamiltonian(kxi, kyi, alpha, t) 

        eigvals_tmp, eigvecs_tmp = np.linalg.eigh(H)
        E[ix, iy, :] = eigvals_tmp[:] * ry2ev
        eigvecs[ix, iy, :, :] = eigvecs_tmp

        # occupation 수 계산 (Fermi-Dirac 분포, T=0)
        occ[ix, iy, :] = 1.0 / (1.0 + np.exp((E[ix, iy, :] - 0.0) / (0.01 * ry2ev)))  # 화학 퍼텐셜 0.0 eV

print("Energy bands calculated.")

# 스핀 기대값 저장용 배열
Sx = np.zeros((nky, nkx, 2))
Sy = np.zeros((nky, nkx, 2))
Sz = np.zeros((nky, nkx, 2))

sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

for ix in range(nkx):
    for iy in range(nky):
        psi = eigvecs[ix, iy, :, 0]  # 낮은 밴드의 고유벡터
        Sx[ix, iy, 0] = np.vdot(psi, sigma_x @ psi).real
        Sy[ix, iy, 0] = np.vdot(psi, sigma_y @ psi).real
        Sz[ix, iy, 0] = np.vdot(psi, sigma_z @ psi).real
        psi = eigvecs[iy, ix, :, 1]  # 낮은 밴드의 고유벡터
        Sx[ix, iy, 1] = np.vdot(psi, sigma_x @ psi).real
        Sy[ix, iy, 1] = np.vdot(psi, sigma_y @ psi).real
        Sz[ix, iy, 1] = np.vdot(psi, sigma_z @ psi).real

print("Spin expectation values calculated.")
print("Sx:", np.sum(Sx[:,:,0] * occ[:,:,0]), np.sum(Sx[:,:,1] * occ[:,:,1]))
print("Sy:", np.sum(Sy[:,:,0] * occ[:,:,0]), np.sum(Sy[:,:,1] * occ[:,:,1]))
print("Sz:", np.sum(Sz[:,:,0] * occ[:,:,0]), np.sum(Sz[:,:,1] * occ[:,:,1]))

# 시각화
# kx=0인 line을 따라 에너지 밴드 시각화 (spin texture color, E± 겹쳐서)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
ky0_idx = np.argmin(np.abs(ky))  # ky=0에 가장 가까운 인덱스

# Sx
sc0 = axes[0].scatter(kx, E[:, ky0_idx, 0], c=Sx[:, ky0_idx, 0], cmap='coolwarm', marker='o', label='E-', vmin=-0.5, vmax=0.5)
sc0_2 = axes[0].scatter(kx, E[:, ky0_idx, 1], c=Sx[:, ky0_idx, 1], cmap='coolwarm', marker='o', label='E+', vmin=-0.5, vmax=0.5)
axes[0].set_title('ky=0, color: Sx')
axes[0].set_xlabel('kx')
axes[0].set_ylabel('Energy')
cbar0 = fig.colorbar(sc0, ax=axes[0], label='Sx')
axes[0].legend(['E-', 'E+'])

# Sy
sc1 = axes[1].scatter(kx, E[:, ky0_idx, 0], c=Sy[:, ky0_idx, 0], cmap='coolwarm', marker='o', label='E-', vmin=-0.5, vmax=0.5)
sc1_2 = axes[1].scatter(kx, E[:, ky0_idx, 1], c=Sy[:, ky0_idx, 1], cmap='coolwarm', marker='o', label='E+', vmin=-0.5, vmax=0.5)
axes[1].set_title('ky=0, color: Sy')
axes[1].set_xlabel('kx')
axes[1].set_ylabel('Energy')
cbar1 = fig.colorbar(sc1, ax=axes[1], label='Sy')
axes[1].legend(['E-', 'E+'])

# Sz
sc2 = axes[2].scatter(kx, E[:, ky0_idx, 0], c=Sz[:, ky0_idx, 0], cmap='coolwarm', marker='o', label='E-', vmin=-0.5, vmax=0.5)
sc2_2 = axes[2].scatter(kx, E[:, ky0_idx, 1], c=Sz[:, ky0_idx, 1], cmap='coolwarm', marker='o', label='E+', vmin=-0.5, vmax=0.5)
axes[2].set_title('ky=0, color: Sz')
axes[2].set_xlabel('kx')
axes[2].set_ylabel('Energy')
cbar2 = fig.colorbar(sc2, ax=axes[2], label='Sz')
axes[2].legend(['E-', 'E+'])

plt.suptitle('Rashba Band (E±) along kx=0 with Spin Texture (Overlapped)')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# 2D spin texture 시각화 (Band 0: blue, Band 1: red, Sz 미표시)
fig2, ax2 = plt.subplots(1, 1, figsize=(7, 6))
step = 4  # 벡터 간격 (coarse하게)

# Band 0 (blue)
X0 = KX.T[::step, ::step]
Y0 = KY.T[::step, ::step]
U0 = Sx[:, :, 0].T[::step, ::step]
V0 = Sy[:, :, 0].T[::step, ::step]
q0 = ax2.quiver(X0, Y0, U0, V0, color='blue', scale=30, width=0.003, label='Band 0')

# Band 1 (red)
# U1 = Sx[:, :, 1].T[::step, ::step]
# V1 = Sy[:, :, 1].T[::step, ::step]
# q1 = ax2.quiver(X0, Y0, U1, V1, color='red', scale=30, width=0.003, label='Band 1')

ax2.set_title('2D Spin Texture: Sx/Sy (Band 0: blue, Band 1: red)')
ax2.set_xlabel('kx')
ax2.set_ylabel('ky')
ax2.quiverkey(q0, X=0.85, Y=1.05, U=1, label='Band 0', labelpos='E', color='blue')
# ax2.quiverkey(q1, X=0.85, Y=1.00, U=1, label='Band 1', labelpos='E', color='red')
plt.tight_layout()
plt.show()

# 두 밴드의 Sx, Sy 합을 quiver로 시각화
fig3, ax3 = plt.subplots(1, 1, figsize=(7, 6))
step = 4
X = KX.T[::step, ::step]
Y = KY.T[::step, ::step]
U_sum = (Sx[:, :, 0] + Sx[:, :, 1]).T[::step, ::step]
V_sum = (Sy[:, :, 0] + Sy[:, :, 1]).T[::step, ::step]
q_sum = ax3.quiver(X, Y, U_sum, V_sum, color='purple', scale=30, width=0.003, label='Band 0 + Band 1')
ax3.set_title('2D Spin Texture: Sx/Sy (Band 0 + Band 1, purple)')
ax3.set_xlabel('kx')
ax3.set_ylabel('ky')
ax3.quiverkey(q_sum, X=0.85, Y=1.05, U=1, label='Sum', labelpos='E', color='purple')
plt.tight_layout()
plt.show()

def td_E_fld(tinfs):
    # 시간에 따른 전기장 계산 (예시로 단순한 사인 함수 사용)
    E0 = 0.0005 / ry2VpA  # 전기장 강도
    dc_st = 10 / ry2fs # 시간 상수
    
    dc_amp = np.zeros(3, dtype=float)
    dc_E_amp = np.zeros(3, dtype=float)
    td_dc_amp = np.array([E0, 0.0, 0.0])  # z 방향 전기장
    if tinfs < dc_st:
        dc_amp[:] = -td_dc_amp[:] * 2.0 / fine_const * (tinfs**3 / dc_st**2 - 0.5 * tinfs**4 / dc_st**3)
        dc_E_amp[:] = td_dc_amp[:] * (3.0 * (tinfs / dc_st)**2 - 2.0 * (tinfs / dc_st)**3)

    elif dc_st < tinfs:
        dc_amp[:] = -td_dc_amp[:] * 2.0 / fine_const * (tinfs - 0.5 * dc_st)
        dc_E_amp[:] = td_dc_amp[:]

    A = dc_amp
    E = dc_E_amp
    
    return A, E

def U_op(H):
    I = np.eye(H.shape[0], dtype=complex)
    A = I - 1j * H * dt / (2 * hbar)
    B = I + 1j * H * dt / (2 * hbar)
    U = np.linalg.inv(A) @ B
    return U

def rashba_td_hamiltonian(kx, ky, A, lam, t=1.0):
    # Identity and Pauli matrices
    I = np.eye(2)
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])

    # Kinetic term
    eps_k = -2 * t * (np.cos(kx+((e/c)*A[0])) + np.cos(ky+((e/c)*A[1]))) * I

    # Rashba SOC term
    rashba_term = 2 * lam * (np.sin(ky+((e/c)*A[1])) * sigma_x - np.sin(kx+((e/c)*A[0])) * sigma_y)

    H = eps_k + rashba_term
    return H

# 시간에 따른 total Sx, Sy, Sz 저장용 리스트
Sx_total = []
Sy_total = []
Sz_total = []

# 초기 상태 설정
eigvecs_td = np.zeros((nky, nkx, 2, 2), dtype=complex)
# 초기 상태는 고유벡터로 설정
for ix in range(nkx):
    for iy in range(nky):
        eigvecs_td[ix, iy, :, :] = eigvecs[ix, iy, :, :]

for istep in range(timestep-1):
    # time evolution
    tinry = (istep+1) * dt
    tinfs = tinry * ry2fs
    A = np.zeros(3)
    A, E_fld = td_E_fld(tinfs)
    for ix in range(nkx):
        for iy in range(nky):
            kxi = KX[ix, iy]
            kyi = KY[ix, iy]
            H_t = rashba_td_hamiltonian(kxi, kyi, A, alpha, t)
            U = U_op(H_t) 

            # 시간에 따른 고유벡터 업데이트
            psi_t = eigvecs_td[ix, iy, :, :]
            psi_t = U @ psi_t
            eigvecs_td[ix, iy, :, :] = psi_t

            if (istep + 1) % 10 == 0:
                Sx[ix, iy, 0] = np.vdot(psi_t[:, 0], sigma_x @ psi_t[:, 0]).real
                Sy[ix, iy, 0] = np.vdot(psi_t[:, 0], sigma_y @ psi_t[:, 0]).real
                Sz[ix, iy, 0] = np.vdot(psi_t[:, 0], sigma_z @ psi_t[:, 0]).real
                Sx[ix, iy, 1] = np.vdot(psi_t[:, 1], sigma_x @ psi_t[:, 1]).real
                Sy[ix, iy, 1] = np.vdot(psi_t[:, 1], sigma_y @ psi_t[:, 1]).real
                Sz[ix, iy, 1] = np.vdot(psi_t[:, 1], sigma_z @ psi_t[:, 1]).real

            if (istep+1) % (timestep / 10) == 0:
                E[ix, iy, 0] = np.vdot(psi_t[:, 0], H_t @ psi_t[:, 0]).real * ry2ev
                E[ix, iy, 1] = np.vdot(psi_t[:, 1], H_t @ psi_t[:, 1]).real * ry2ev
                
    if (istep+1) % 10 == 0:
        print(f"Step {istep+1}, Time: {tinfs:10.6e}fs")
        print(f"E-field: {E_fld[0]*ry2VpA:.6e} V/A, {E_fld[1]*ry2VpA:.6e} V/A, {E_fld[2]*ry2VpA:.6e} V/A")
        print(f"A-field: {A[0]:.6e}, {A[1]:.6e}, {A[2]:.6e}")
        
    # if (istep+1) % (timestep / 10) == 0:
    #     # 시각화
    #     # kx=0인 line을 따라 에너지 밴드 시각화 (spin texture color, E± 겹쳐서)
    #     fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    #     ky0_idx = np.argmin(np.abs(ky))  # ky=0에 가장 가까운 인덱스

    #     # Sx
    #     sc0 = axes[0].scatter(kx, E[:, ky0_idx, 0], c=Sx[:, ky0_idx, 0], cmap='coolwarm', marker='o', label='E-', vmin=-0.5, vmax=0.5)
    #     sc0_2 = axes[0].scatter(kx, E[:, ky0_idx, 1], c=Sx[:, ky0_idx, 1], cmap='coolwarm', marker='o', label='E+', vmin=-0.5, vmax=0.5)
    #     axes[0].set_title('ky=0, color: Sx')
    #     axes[0].set_xlabel('kx')
    #     axes[0].set_ylabel('Energy')
    #     cbar0 = fig.colorbar(sc0, ax=axes[0], label='Sx')
    #     axes[0].legend(['E-', 'E+'])

    #     # Sy
    #     sc1 = axes[1].scatter(kx, E[:, ky0_idx, 0], c=Sy[:, ky0_idx, 0], cmap='coolwarm', marker='o', label='E-', vmin=-0.5, vmax=0.5)
    #     sc1_2 = axes[1].scatter(kx, E[:, ky0_idx, 1], c=Sy[:, ky0_idx, 1], cmap='coolwarm', marker='o', label='E+', vmin=-0.5, vmax=0.5)
    #     axes[1].set_title('ky=0, color: Sy')
    #     axes[1].set_xlabel('kx')
    #     axes[1].set_ylabel('Energy')
    #     cbar1 = fig.colorbar(sc1, ax=axes[1], label='Sy')
    #     axes[1].legend(['E-', 'E+'])

    #     # Sz
    #     sc2 = axes[2].scatter(kx, E[:, ky0_idx, 0], c=Sz[:, ky0_idx, 0], cmap='coolwarm', marker='o', label='E-', vmin=-0.5, vmax=0.5)
    #     sc2_2 = axes[2].scatter(kx, E[:, ky0_idx, 1], c=Sz[:, ky0_idx, 1], cmap='coolwarm', marker='o', label='E+', vmin=-0.5, vmax=0.5)
    #     axes[2].set_title('ky=0, color: Sz')
    #     axes[2].set_xlabel('kx')
    #     axes[2].set_ylabel('Energy')
    #     cbar2 = fig.colorbar(sc2, ax=axes[2], label='Sz')
    #     axes[2].legend(['E-', 'E+'])

    #     plt.suptitle('Rashba Band (E±) along kx=0 with Spin Texture (Overlapped)')
    #     plt.tight_layout(rect=[0, 0, 1, 0.96])
    #     plt.show()

    if (istep + 1) % 10 == 0:
        Sx_sum0 = np.sum(Sx[:,:,0] * occ[:,:,0])
        Sx_sum1 = np.sum(Sx[:,:,1] * occ[:,:,1])
        Sy_sum0 = np.sum(Sy[:,:,0] * occ[:,:,0])
        Sy_sum1 = np.sum(Sy[:,:,1] * occ[:,:,1])
        Sz_sum0 = np.sum(Sz[:,:,0] * occ[:,:,0])
        Sz_sum1 = np.sum(Sz[:,:,1] * occ[:,:,1])
        print(f"Step {istep+1}, Time: {tinfs:10.6e}fs")
        print(f"Sx-: {Sx_sum0}, Sx+: {Sx_sum1}")
        print(f"Sy-: {Sy_sum0}, Sy+: {Sy_sum1}")
        print(f"Sz-: {Sz_sum0}, Sz+: {Sz_sum1}")
        Sx_total.append([tinry, Sx_sum0, Sx_sum1])
        Sy_total.append([tinry, Sy_sum0, Sy_sum1])
        Sz_total.append([tinry, Sz_sum0, Sz_sum1])

# 시뮬레이션 종료 후 파일로 저장
Sx_total = np.array(Sx_total)
Sy_total = np.array(Sy_total)
Sz_total = np.array(Sz_total)
np.savetxt('Sx_total.txt', Sx_total, header='time Sx- Sx+', fmt='%.8e')
np.savetxt('Sy_total.txt', Sy_total, header='time Sy- Sy+', fmt='%.8e')
np.savetxt('Sz_total.txt', Sz_total, header='time Sz- Sz+', fmt='%.8e')
print("Total spin expectation values saved to Sx_total.txt, Sy_total.txt, Sz_total.txt")
print("Time evolution completed.")