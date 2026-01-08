"""
Monte Carlo Simulation für Statistische Physik Bonusübung
Vollständig optimiert und korrigierte Version
"""

import numpy as np
import matplotlib.pyplot as plt

# Set matplotlib to use proper font for symbols
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'


class Paramagnet:
    """Optimierte Monte-Carlo-Simulation nicht-wechselwirkender Spins."""
    
    def __init__(self, L, h=1.0):
        self.L = L
        self.h = h
        self.spins = np.random.choice([-1, 1], size=L).astype(np.int8)
    
    def metropolis_update_batch(self, N, beta):
        """Führt N Metropolis-Updates effizient aus."""
        indices = np.random.randint(0, self.L, size=N, dtype=np.int32)
        randoms = np.random.random(size=N)
        two_h = 2.0 * self.h
        
        for idx, r in zip(indices, randoms):
            delta_E = two_h * self.spins[idx]
            if delta_E <= 0.0 or r < np.exp(-beta * delta_E):
                self.spins[idx] *= -1
    
    def simulate_chain(self, N, beta, initial_config=None):
        """Simuliert Markov-Kette der Länge N Metropolis-Updates."""
        if initial_config is not None:
            self.spins = initial_config.copy().astype(np.int8)
        else:
            self.spins = np.random.choice([-1, 1], size=self.L).astype(np.int8)
        
        self.metropolis_update_batch(N, beta)
        return np.mean(self.spins, dtype=np.float64)
    
    def thermalize_and_measure(self, N_therm, N_mc, k, beta):
        """Thermalisiert und misst effizient."""
        self.metropolis_update_batch(N_therm * self.L, beta)
        measurements = np.empty(N_mc, dtype=np.float64)
        
        for i in range(N_mc):
            self.metropolis_update_batch(k * self.L, beta)
            measurements[i] = np.mean(self.spins, dtype=np.float64)
        
        return measurements


class IsingModel:
    """Hochoptimierte Monte-Carlo-Simulation des 2D-Ising-Modells."""
    
    def __init__(self, L, J=1.0):
        self.L = L
        self.J = J
        self.spins = np.random.choice([-1, 1], size=(L, L)).astype(np.int8)
        self.L2 = L * L
        self.two_J = 2.0 * J
    
    def metropolis_update_batch(self, N, beta):
        """Führt N Metropolis-Updates effizient aus - optimiert."""
        L = self.L
        chunk_size = min(10000, N)
        total_updates = 0
        
        while total_updates < N:
            remaining = N - total_updates
            current_chunk = min(chunk_size, remaining)
            
            coords_i = np.random.randint(0, L, size=current_chunk, dtype=np.int32)
            coords_j = np.random.randint(0, L, size=current_chunk, dtype=np.int32)
            randoms = np.random.random(size=current_chunk)
            
            for i, j, r in zip(coords_i, coords_j, randoms):
                neighbor_sum = (self.spins[(i+1) % L, j] +
                               self.spins[(i-1) % L, j] +
                               self.spins[i, (j+1) % L] +
                               self.spins[i, (j-1) % L])
                delta_E = self.two_J * self.spins[i, j] * neighbor_sum
                
                if delta_E <= 0.0 or r < np.exp(-beta * delta_E):
                    self.spins[i, j] *= -1
            
            total_updates += current_chunk
    
    def thermalize_and_measure(self, N_therm, N_mc, k, beta, measure_abs_M=True):
        """Thermalisiert und misst effizient."""
        L2 = self.L2
        self.metropolis_update_batch(N_therm * L2, beta)
        
        M_measurements = np.empty(N_mc, dtype=np.float64)
        M2_measurements = np.empty(N_mc, dtype=np.float64)
        
        for i in range(N_mc):
            self.metropolis_update_batch(k * L2, beta)
            M = np.mean(self.spins, dtype=np.float64)
            M_measurements[i] = np.abs(M) if measure_abs_M else M
            M2_measurements[i] = M * M
        
        return M_measurements, M2_measurements


def part1a():
    """Teil 1a: Magnetisierung vs. Kettenlänge."""
    print("Teil 1a: Magnetisierung vs. Kettenlänge")
    
    L = 100
    beta = 1.0
    h = 1.0
    N_values = [10, 50, 100, 250, 500, 750, 1000]
    
    paramagnet = Paramagnet(L, h)
    magnetizations = []
      
    for N in N_values:
        paramagnet.spins = np.random.choice([-1, 1], size=L).astype(np.int8)
        M_final = paramagnet.simulate_chain(N, beta)
        magnetizations.append(M_final)
        print(f"N = {N:4d}, M = {M_final:.6f}")
    
    M_analytical = np.tanh(beta * h)
    
    fig = plt.figure(figsize=(12, 5))
    fig.suptitle('Bonusübung - Statistische Mechanik: Monte-Carlo-Simulationen', fontsize=14, fontweight='bold')
    
    ax1 = fig.add_subplot(121)
    ax1.plot([1/N for N in N_values], magnetizations, 'o-', label='Numerisch')
    ax1.axhline(y=M_analytical, color='r', linestyle='--', label=f'Analytisch = {M_analytical:.6f}')
    ax1.set_xlabel('1/N')
    ax1.set_ylabel(r'$\langle M \rangle$')
    ax1.set_title('Teil 1a: Konvergenz der Magnetisierung')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(122)
    ax2.plot(N_values, magnetizations, 'o-', label='Numerisch')
    ax2.axhline(y=M_analytical, color='r', linestyle='--', label=f'Analytisch = {M_analytical:.6f}')
    ax2.set_xlabel('Kettenlänge N')
    ax2.set_ylabel(r'$\langle M \rangle$')
    ax2.set_title('Teil 1a: $\langle M \\rangle$ abhängig von N')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('part1a_magnetization_vs_chain_length.png', dpi=150)
    plt.close()
    
    return N_values, magnetizations


def part1b():
    """Teil 1b: Statistische Schwankungen vs. L und N_MC."""
    print("\nTeil 1b: Statistische Schwankungen")
    
    beta = 1.0
    h = 1.0
    N_therm = 50
    k = 3
    
    print("\n(i) Festes N_MC = 100, variierendes L")
    N_MC_fixed = 100
    L_values = [20, 50, 100, 200, 500]
    
    means_L = []
    stds_L = []
    
    for L in L_values:
        paramagnet = Paramagnet(L, h)
        measurements = paramagnet.thermalize_and_measure(N_therm, N_MC_fixed, k, beta)
        means_L.append(np.mean(measurements))
        stds_L.append(np.std(measurements, ddof=1))
        print(f"L = {L:4d}, ⟨M⟩ = {means_L[-1]:.6f}, σ = {stds_L[-1]:.6f}")
    
    print("\n(ii) Festes L = 100, variierendes N_MC")
    L_fixed = 100
    N_MC_values = [10, 50, 100, 250, 500, 1000, 2500]
    
    means_N = []
    stds_N = []
    
    for N_MC in N_MC_values:
        paramagnet = Paramagnet(L_fixed, h)
        measurements = paramagnet.thermalize_and_measure(N_therm, N_MC, k, beta)
        means_N.append(np.mean(measurements))
        stds_N.append(np.std(measurements, ddof=1))
        print(f"N_MC = {N_MC:4d}, ⟨M⟩ = {means_N[-1]:.6f}, σ = {stds_N[-1]:.6f}")
    
    M_analytical = np.tanh(beta * h)
    
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('Bonusübung - Statistische Mechanik: Monte-Carlo-Simulationen', fontsize=14, fontweight='bold')
    
    ax1 = fig.add_subplot(221)
    ax1.plot(L_values, means_L, 'o-', label='Numerisch')
    ax1.axhline(y=M_analytical, color='r', linestyle='--', label=f'Analytisch = {M_analytical:.6f}')
    ax1.set_xlabel('Systemgröße L')
    ax1.set_ylabel(r'$\langle M \rangle$')
    ax1.set_title('Teil 1b(i): $\langle M \\rangle$ abhängig von L')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(222)
    ax2.plot(L_values, stds_L, 'o-', color='orange', markersize=6)
    ax2.set_xlabel('Systemgröße L')
    ax2.set_ylabel(r'$\sigma$')
    ax2.set_title('Teil 1b(i): $\sigma$ abhängig von L')
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(223)
    ax3.plot(N_MC_values, means_N, 'o-', label='Numerisch')
    ax3.axhline(y=M_analytical, color='r', linestyle='--', label=f'Analytisch = {M_analytical:.6f}')
    ax3.set_xlabel(r'Anzahl der Messungen $N_{MC}$')
    ax3.set_ylabel(r'$\langle M \rangle$')
    ax3.set_title('Teil 1b(ii): $\langle M \\rangle$ abhängig von $N_{MC}$')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    
    ax4 = fig.add_subplot(224)
    ax4.plot(N_MC_values, stds_N, 'o-', color='orange', markersize=6)
    ax4.set_xlabel(r'Anzahl der Messungen $N_{MC}$')
    ax4.set_ylabel(r'$\sigma$')
    ax4.set_title('Teil 1b(ii): $\sigma$ abhängig von $N_{MC}$')
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('part1b_statistical_fluctuations.png', dpi=150)
    plt.close()
    
    return L_values, means_L, stds_L, N_MC_values, means_N, stds_N


def part1c():
    """Teil 1c: Magnetisierung und Suszeptibilität vs. Temperatur."""
    print("\nTeil 1c: Magnetisierung und Suszeptibilität vs. Temperatur")
    
    L = 100
    h = 1.0
    N_therm = 100
    N_MC = 2000
    k = 3
    
    T_values = np.linspace(0.5, 5.0, 30)
    beta_values = 1.0 / T_values
    
    M_values = np.empty(len(T_values))
    M_errors = np.empty(len(T_values))
    chi_values = np.empty(len(T_values))
    chi_errors = np.empty(len(T_values))
    
    paramagnet = Paramagnet(L, h)
    
    for idx, (T, beta) in enumerate(zip(T_values, beta_values)):
        paramagnet.spins = np.random.choice([-1, 1], size=L).astype(np.int8)
        measurements = paramagnet.thermalize_and_measure(N_therm, N_MC, k, beta)
        
        M_mean = np.mean(measurements)

        # --- Block-Averaging für Chi ---
        num_blocks = 20
        block_size = N_MC // num_blocks
        chi_blocks = np.empty(num_blocks)
        
        for b in range(num_blocks):
            block = measurements[b * block_size : (b + 1) * block_size]
            m_block = np.mean(block)
            m2_block = np.mean(block**2)
            # χ = β(⟨M²⟩ - ⟨M⟩²) 
            chi_blocks[b] = beta * (m2_block - m_block**2)
            
        chi = np.mean(chi_blocks)
        chi_error = np.std(chi_blocks, ddof=1) / np.sqrt(num_blocks)
        
        # M_std_err sollte ebenfalls über Blöcke berechnet werden für Konsistenz
        m_blocks = [np.mean(measurements[b*block_size:(b+1)*block_size]) for b in range(num_blocks)]
        M_std_err = np.std(m_blocks, ddof=1) / np.sqrt(num_blocks)
        
        M_values[idx] = M_mean
        M_errors[idx] = M_std_err
        chi_values[idx] = chi
        chi_errors[idx] = chi_error
        
        if (idx + 1) % 5 == 0:
            print(f"T = {T:.2f}, ⟨M⟩ = {M_mean:.6f} +/- {M_std_err:.6f}, χ = {chi:.6f} +/- {chi_error:.6f}")
    
    # Analytische Ergebnisse
    M_analytical = np.tanh(beta_values * h)
    chi_analytical_total = beta_values / np.cosh(beta_values * h)**2
    chi_analytical_per_spin = chi_analytical_total / L
    
    fig = plt.figure(figsize=(14, 5))
    fig.suptitle('Bonusübung - Statistische Mechanik: Monte-Carlo-Simulationen', fontsize=14, fontweight='bold')
    
    ax1 = fig.add_subplot(121)
    ax1.errorbar(T_values, M_values, yerr=M_errors, fmt='o', capsize=4, label='Numerisch', markersize=5, elinewidth=2)
    ax1.plot(T_values, M_analytical, 'r--', linewidth=2, label='Analytisch')
    ax1.set_xlabel('Temperatur T')
    ax1.set_ylabel(r'$\langle M \rangle$')
    ax1.set_title('Teil 1c: $\langle M \\rangle$ abhängig von T')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(122)
    ax2.errorbar(T_values, chi_values, yerr=chi_errors, fmt='o', capsize=4, label='Numerisch (pro Spin)', markersize=5, elinewidth=2)
    ax2.plot(T_values, chi_analytical_per_spin, 'r--', linewidth=2, label=f'Analytisch (pro Spin, L={L})')
    ax2.set_xlabel('Temperatur T')
    ax2.set_ylabel(r'$\chi$ (pro Spin)')
    ax2.set_title('Teil 1c: $\chi$ abhängig von T')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('part1c_magnetization_susceptibility.png', dpi=150)
    plt.close()
    
    print("\nTeil 1c abgeschlossen.")
    
    return T_values, M_values, M_errors, chi_values, chi_errors


def part2a():
    """Teil 2a: Ising-Modell Magnetisierung vs. Temperatur."""
    print("\nTeil 2a: Ising-Modell |M| vs. Temperatur")
    
    J = 1.0
    L_values = [20, 30]
    T_values = np.linspace(1.0, 4.0, 25)
    N_therm = 1000
    N_MC = 10000
    k = 3
    
    T_c = 2.0 / np.log(1.0 + np.sqrt(2.0))
    
    results = {}
    
    for L in L_values:
        print(f"\nL = {L}")
        ising = IsingModel(L, J)
        M_values = np.empty(len(T_values))
        M_errors = np.empty(len(T_values))
        
        for idx, T in enumerate(T_values):
            beta = 1.0 / T
            ising.spins = np.random.choice([-1, 1], size=(L, L)).astype(np.int8)
            M_measurements, _ = ising.thermalize_and_measure(N_therm, N_MC, k, beta, measure_abs_M=True)
            
            M_mean = np.mean(M_measurements)
            M_std_err = np.std(M_measurements, ddof=1) / np.sqrt(N_MC)
            
            M_values[idx] = M_mean
            M_errors[idx] = M_std_err
            
            if (idx + 1) % 5 == 0:
                print(f"  T = {T:.2f}, |⟨M⟩| = {M_mean:.6f} +/- {M_std_err:.6f}")
        
        results[L] = (M_values, M_errors)
    
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle('Bonusübung - Statistische Mechanik: Monte-Carlo-Simulationen', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    
    for L in L_values:
        M_vals, M_errs = results[L]
        ax.errorbar(T_values, M_vals, yerr=M_errs, fmt='o-', capsize=4, label=f'L = {L}', markersize=5, elinewidth=2)
    
    ax.axvline(x=T_c, color='r', linestyle='--', linewidth=2, label=f'$T_c$ = {T_c:.4f}')
    ax.set_xlabel('Temperatur T')
    ax.set_ylabel(r'$|\langle M \rangle|$')
    ax.set_title('Teil 2a: $|\langle M \\rangle|$ abhängig von T')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('part2a_ising_magnetization.png', dpi=150)
    plt.close()
    
    return T_values, results, T_c


def part2b():
    """Teil 2b: Visualisierung Gleichgewichtskonfigurationen."""
    print("\nTeil 2b: Visualisierung Gleichgewichtskonfigurationen")
    
    L = 50
    J = 1.0
    T_c = 2.0 / np.log(1.0 + np.sqrt(2.0))
    
    temperatures = [1.5, T_c, 3.0]
    labels = [f'T = {1.5:.2f} < T_c', f'T = {T_c:.4f} ≈ T_c', f'T = {3.0:.2f} > T_c']
    
    N_therm = 1000
    N_MC = 1
    k = 1
    
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle('Bonusübung - Statistische Mechanik: Monte-Carlo-Simulationen', fontsize=14, fontweight='bold')
    
    for idx, (T, label) in enumerate(zip(temperatures, labels)):
        beta = 1.0 / T
        ising = IsingModel(L, J)
        ising.thermalize_and_measure(N_therm, N_MC, k, beta, measure_abs_M=True)
        
        ax = fig.add_subplot(1, 3, idx + 1)
        ax.imshow(ising.spins, cmap='RdYlBu', vmin=-1, vmax=1, interpolation='nearest')
        ax.set_title(f'Teil 2b: {label.replace("T_c", "$T_c$")}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('part2b_ising_configurations.png', dpi=150)
    plt.close()
    
    print("Konfigurationen gespeichert für T < T_c, T ≈ T_c und T > T_c")


def part2c():
    """Teil 2c: Ising-Modell Suszeptibilität vs. Temperatur."""
    print("\nTeil 2c: Ising-Modell Suszeptibilität vs. Temperatur")
    
    L = 30
    J = 1.0
    T_values = np.linspace(1.5, 3.5, 30)
    N_therm = 1000
    N_MC = 10000
    k = 1
    
    T_c = 2.0 / np.log(1.0 + np.sqrt(2.0))
    
    chi_values = np.empty(len(T_values))
    chi_errors = np.empty(len(T_values))
    
    ising = IsingModel(L, J)
    
    for idx, T in enumerate(T_values):
        try:
            beta = 1.0 / T
            ising.spins = np.random.choice([-1, 1], size=(L, L)).astype(np.int8)
            
            if (idx + 1) % 5 == 0 or idx == 0 or idx == len(T_values) - 1:
                print(f"Berechne T = {T:.2f} ({idx+1}/{len(T_values)})...")
            
            M_measurements, _ = ising.thermalize_and_measure(N_therm, N_MC, k, beta, measure_abs_M=False)
            
            # M_measurements ist dein Array mit N_MC Werten
            num_blocks = 20
            block_size = N_MC // num_blocks
            chi_blocks = []

            for b in range(num_blocks):
                block = M_measurements[b*block_size : (b+1)*block_size]
                # Chi für diesen spezifischen Block berechnen
                m_block = np.mean(block)
                m2_block = np.mean(block**2)
                chi_blocks.append(beta * (m2_block - m_block**2))

            # Der finale Wert ist der Mittelwert der Block-Chis
            chi = np.mean(chi_blocks)
            # Der Fehler ist der Standardfehler der Block-Mittelwerte
            chi_error = np.std(chi_blocks, ddof=1) / np.sqrt(num_blocks)
            
            chi_values[idx] = chi
            chi_errors[idx] = chi_error
            
            if (idx + 1) % 5 == 0 or idx == 0 or idx == len(T_values) - 1:
                print(f"T = {T:.2f}, χ = {chi:.6f} +/- {chi_error:.6f}")
        except Exception as e:
            print(f"Fehler bei T = {T:.2f}: {e}")
            import traceback
            traceback.print_exc()
            chi_values[idx] = np.nan
            chi_errors[idx] = np.nan
    
    # Überprüfen, dass alle Werte gesetzt wurden
    print(f"\nGesamt berechnet: {np.sum(~np.isnan(chi_values))}/{len(T_values)} Werte")
    
    # NaN-Werte für Plot entfernen
    valid_mask = ~np.isnan(chi_values)
    if np.sum(valid_mask) == 0:
        print("FEHLER: Keine gültigen Daten für Plot!")
        return T_values, chi_values, chi_errors, T_c
    
    T_plot = T_values[valid_mask]
    chi_plot = chi_values[valid_mask]
    chi_err_plot = chi_errors[valid_mask]
    
    # Sicherstellen, dass Arrays die richtige Länge haben
    assert len(T_plot) == len(chi_plot) == len(chi_err_plot), f"Längen stimmen nicht überein: T={len(T_plot)}, chi={len(chi_plot)}, err={len(chi_err_plot)}"
    
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle('Bonusübung - Statistische Mechanik: Monte-Carlo-Simulationen', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111)
    
    ax.errorbar(T_plot, chi_plot, yerr=chi_err_plot, fmt='o-', capsize=4, markersize=5, elinewidth=2)
    ax.axvline(x=T_c, color='r', linestyle='--', linewidth=2, label=f'$T_c$ = {T_c:.4f}')
    ax.set_xlabel('Temperatur T')
    ax.set_ylabel(r'$\chi$')
    ax.set_title(f'Teil 2c: $\chi$ abhängig von T (L = {L})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('part2c_ising_susceptibility.png', dpi=150)
    plt.close()
    
    if np.sum(valid_mask) > 0:
        print(f"\nTeil 2c abgeschlossen. χ-Bereich: {np.nanmin(chi_values[valid_mask]):.6f} bis {np.nanmax(chi_values[valid_mask]):.6f}")
        print(f"Plottete {len(T_plot)} von {len(T_values)} Werten")
    else:
        print("\nFEHLER: Keine gültigen Daten zum Plotten!")
    
    return T_values, chi_values, chi_errors, T_c


if __name__ == "__main__":
    print("Monte-Carlo-Simulation - Statistische Physik Bonusübung")
    print("=" * 60)
    
    # Teil 1
    print("\n" + "=" * 60)
    print("TEIL 1: Nicht-wechselwirkende Spins (Paramagnet)")
    print("=" * 60)
    
    try:
        part1a()
        part1b()
        part1c()
    except Exception as e:
        print(f"Fehler in Teil 1: {e}")
        import traceback
        traceback.print_exc()
    
    # Teil 2
    # print("\n" + "=" * 60)
    # print("TEIL 2: Ising-Modell auf dem Quadratgitter")
    # print("=" * 60)
    
    # try:
    #     part2a()
    #     part2b()
    #     part2c()
    # except Exception as e:
    #     print(f"Fehler in Teil 2: {e}")
    #     import traceback
    #     traceback.print_exc()
    
    # print("\n" + "=" * 60)
    # print("Alle Simulationen abgeschlossen!")
    # print("=" * 60)
