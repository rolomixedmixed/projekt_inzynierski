import cupy as cp
from cuquantum import tensornet as tn

print("--- Budowa bramki Fredkina w cuQuantum ---")

# Funkcja pomocnicza: tworzy czystą macierz 8x8 (brak zmian na kubitach)
def identity_8x8():
    return cp.eye(8, dtype=cp.complex128)

# ---------------------------------------------------------
# KROK 1: Definiowanie bramek Toffoliego (macierze 8x8)
# ---------------------------------------------------------

# BRAMKA 1: Kontrola c i t1, Cel t2.
# Działa tylko gdy c=1 i t1=1. Zamienia miejscami |110> (indeks 6) i |111> (indeks 7)
T1 = identity_8x8()
T1[6, 6] = 0; T1[6, 7] = 1
T1[7, 6] = 1; T1[7, 7] = 0

# BRAMKA 2: Kontrola c i t2, Cel t1.
# Działa tylko gdy c=1 i t2=1. Zamienia miejscami |101> (indeks 5) i |111> (indeks 7)
T2 = identity_8x8()
T2[5, 5] = 0; T2[5, 7] = 1
T2[7, 5] = 1; T2[7, 7] = 0

# BRAMKA 3: Kontrola c i t1, Cel t2 (dokładnie to samo co T1)
T3 = T1.copy()

# ---------------------------------------------------------
# KROK 2: Kontrakcja tensorów (Tworzenie bramki Fredkina)
# ---------------------------------------------------------
# Mnożymy macierze od prawej do lewej (T3 * T2 * T1) używając notacji Einsteina:
# Zamiast 3 operacji, GPU generuje jedną gotową bramkę.
Fredkin = tn.contract('ij,jk,kl->il', T3, T2, T1)

# ---------------------------------------------------------
# KROK 3: Symulacja (Testowanie na wektorze stanu)
# ---------------------------------------------------------
# Chcemy przetestować nasz stan z wcześniejszych rozważań: 
# c=1, t1=1, t2=0, czyli stan |1, 1, 0> (indeks 6)
stan_poczatkowy = cp.zeros(8, dtype=cp.complex128)
stan_poczatkowy[6] = 1.0

print("\nStan początkowy: |1, 1, 0> (indeks 6)")
print(stan_poczatkowy)

# Aplikujemy naszą nową bramkę Fredkina na wektor stanu
stan_koncowy = tn.contract('ij,j->i', Fredkin, stan_poczatkowy)

print("\nStan po bramce Fredkina (3x Toffoli):")
print(stan_koncowy)

# ---------------------------------------------------------
# KROK 4: Weryfikacja wyniku
# ---------------------------------------------------------
# Szukamy, pod którym indeksem znajduje się teraz "1.0"
indeks_wyniku = int(cp.argmax(cp.abs(stan_koncowy)))
stany_binarne = ["000", "001", "010", "011", "100", "101", "110", "111"]

print("-" * 50)
print(f"Oczekiwano: |1, 0, 1>")
print(f"Otrzymano : |{stany_binarne[indeks_wyniku]}> (indeks {indeks_wyniku})")

if stany_binarne[indeks_wyniku] == "101":
    print("✅ SUKCES! Kubity t1 i t2 zostały poprawnie zamienione miejscami!")
else:
    print("❌ Coś poszło nie tak.")