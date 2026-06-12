"""
SRBIN Nikola Tesla, za sva vremena, najveci naucnik sveta.

SERBIAN Nikola Tesla, for all time, the greatest scientist in the world.
"""



"""
Tesla_quantum_wavefield_3B.py  —  GRUPA 3, deo B (talasno polje verovatnoce)

pravi „polje verovatnoce" slaganjem kompleksnih amplitudnih talasa iz
frekvencije, zazora, parova i faznog talasa

srce modela; pravi skorove brojeva

Razlika u odnosu na grupu 1/2: ovde talas NIJE iz fizickog solvera, nego iz
statistike izvlacenja (frekvencija + zazor + parovi) plus sinusna fazna modulacija.
"""


from collections import Counter

import numpy as np
import pandas as pd

from Tesla_Scalar_1 import MIN_BROJ, MAX_BROJ
from Tesla_wave_transform_3A import fazni_talas, oblikuj_amplitudu

# Tezine talasnog modela (iz starog config.py). Menjati jednu po jednu.
TEZINA_FREKVENCIJE = 0.35
TEZINA_ZAZORA = 0.25
TEZINA_PAROVA = 0.20
TEZINA_TALASA = 0.20


def minmax(v):
    """Skaliraj niz na 0..1 (ravno na 0 ako su sve vrednosti jednake)."""
    v = np.asarray(v, dtype=float)
    raspon = v.max() - v.min()
    if raspon <= 0:
        return np.zeros_like(v)
    return (v - v.min()) / raspon


def normalizuj(v):
    """Pretvori u raspodelu verovatnoca (zbir = 1)."""
    v = np.asarray(v, dtype=float)
    v = np.maximum(v, 0)
    s = v.sum()
    if s <= 0:
        return np.ones_like(v) / len(v)
    return v / s


def _frekvencije(izvlacenja):
    c = Counter(b for red in izvlacenja for b in red)
    return np.array([c.get(b, 0) for b in range(MIN_BROJ, MAX_BROJ + 1)], dtype=float)


def _zazor_od_poslednjeg(izvlacenja):
    poslednji = {b: None for b in range(MIN_BROJ, MAX_BROJ + 1)}
    for i, red in enumerate(izvlacenja):
        for b in red:
            poslednji[b] = i
    kraj = len(izvlacenja) - 1
    return np.array([
        len(izvlacenja) if poslednji[b] is None else kraj - poslednji[b]
        for b in range(MIN_BROJ, MAX_BROJ + 1)
    ], dtype=float)


def _parovi(izvlacenja, top_k=6):
    """Jacina najjacih parova preko 'lift'-a (ne prost zbir).

    Prost zbir parova je ~ proporcionalan frekvenciji broja, pa dvaput broji isto.
    Zato merimo koliko par IZLAZI ZAJEDNO vise nego sto bi se ocekivalo po slucaju:
      lift(a,b) = P(a,b) / (P(a) * P(b))
    Za svaki broj uzmemo prosek njegovih top_k najjacih lift-ova (najjaci parovi).
    """
    n = len(izvlacenja)
    c = Counter(b for red in izvlacenja for b in red)
    par_counter = Counter()
    for red in izvlacenja:
        red = list(red)
        for i, a in enumerate(red):
            for b in red[i + 1:]:
                par_counter[tuple(sorted((a, b)))] += 1

    partneri = {b: [] for b in range(MIN_BROJ, MAX_BROJ + 1)}
    for (a, b), v in par_counter.items():
        if c[a] > 0 and c[b] > 0:
            lift = v * n / (c[a] * c[b])
            partneri[a].append(lift)
            partneri[b].append(lift)

    skor = []
    for broj in range(MIN_BROJ, MAX_BROJ + 1):
        lifts = sorted(partneri[broj], reverse=True)[:top_k]
        skor.append(float(np.mean(lifts)) if lifts else 0.0)
    return np.array(skor, dtype=float)


def napravi_talasno_polje(izvlacenja):
    """Napravi 7/39 talasno polje iz frekvencije, zazora, parova i faznog talasa."""
    freq = minmax(_frekvencije(izvlacenja))
    gap = minmax(_zazor_od_poslednjeg(izvlacenja))
    pair = minmax(_parovi(izvlacenja))

    baza = TEZINA_FREKVENCIJE * freq + TEZINA_ZAZORA * gap + TEZINA_PAROVA * pair
    amplituda = oblikuj_amplitudu(baza, metoda="sqrt")
    talas = fazni_talas(amplituda, frekvencija=7, faza=np.pi / 4)
    talasna_snaga = minmax(np.abs(talas) ** 2)

    skor = baza + TEZINA_TALASA * talasna_snaga
    verovatnoca = normalizuj(skor)

    return pd.DataFrame({
        "broj": list(range(MIN_BROJ, MAX_BROJ + 1)),
        "frekvencija_skor": freq,
        "zazor_skor": gap,
        "par_skor": pair,
        "talas_skor": talasna_snaga,
        "skor": skor,
        "verovatnoca": verovatnoca,
    }).sort_values("skor", ascending=False).reset_index(drop=True)



"""
source ~/tesla_env/bin/activate

Bitne verzije za tesla_env:

Paket	Verzija
python  3.11.13
numpy   2.2.6
scipy   1.15.3
pandas  3.0.3
matplotlib    3.10.9
k-Wave-python 0.6.2
pycharge      2.0.1
jax        0.10.1
jaxlib     0.10.1
jaxtyping  0.3.7
equinox    0.13.8
lineax     0.1.1
optimistix 0.1.0
ml-dtypes
(uz jax)
opencv-python 4.13.0.92
h5py          3.16.0
"""
