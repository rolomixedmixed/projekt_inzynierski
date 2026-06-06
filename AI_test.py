import cupy as cp
from cuquantum import tensornet as tn 

print("--- Start obliczeń na GPU ---")

# 1. Definiujemy stan początkowy wymuszając wysoką precyzję (complex128)
stan_poczatkowy = cp.array([0.0 + 0.0j, 1.0 + 0.0j], dtype=cp.complex128)
print("\nStan początkowy wektora |1>:")
print(stan_poczatkowy)

# 2. Definiujemy bramkę Hadamarda (H)
# Dzielenie przez cp.sqrt(2) utrzyma typ complex128
bramka_H = cp.array([[1,  1],
                     [1, -1]], dtype=cp.complex128) / cp.sqrt(2)

# 3. Wykonanie obliczeń za pomocą nowego API tensornet
stan_koncowy = tn.contract('ij,j->i', bramka_H, stan_poczatkowy)

print("\nStan po przejściu przez bramkę H (superpozycja):")
print(stan_koncowy)

# 4. Obliczamy fizyczne prawdopodobieństwo pomiaru
prawdopodobienstwa = cp.abs(stan_koncowy)**2

print("-" * 30)
print(f"Prawdopodobieństwo zmierzenia '0': {prawdopodobienstwa[0] * 100:.1f}%")
print(f"Prawdopodobieństwo zmierzenia '1': {prawdopodobienstwa[1] * 100:.1f}%")