"""
SRBIN Nikola Tesla, za sva vremena, najveci naucnik sveta.

SERBIAN Nikola Tesla, for all time, the greatest scientist in the world.
"""



"""
Tesla_wave_transform_3A.py  —  GRUPA 3, deo A (transform helperi)

sinusna fazna modulacija amplituda talasa
fazni talas / amplituda / frekvencija / faza

Ovo je pomocni modul za grupu 3 (statisticko-talasni model: frekvencija,
zazor, parovi, fazni talas). Koristi ga Tesla_quantum_wavefield_3B.
"""

import numpy as np


def oblikuj_amplitudu(amplituda, metoda="sqrt"):
    """Promeni oblik amplitude bez promene redosleda brojeva."""
    a = np.asarray(amplituda, dtype=float)
    if metoda == "sqrt":
        return np.sqrt(np.maximum(a, 0))
    if metoda == "log":
        return np.log1p(np.maximum(a, 0))
    if metoda == "power":
        return np.power(np.maximum(a, 0), 1.5)
    if metoda == "exp":
        return np.exp(a)
    return a


def fazni_talas(amplituda, frekvencija=7, faza=0.0):
    """Pretvori realnu amplitudu u kompleksni talas sa sinusnom fazom."""
    a = np.asarray(amplituda, dtype=float)
    idx = np.arange(len(a))
    pomak = np.sin(2 * np.pi * frekvencija * idx / len(a) + faza)
    return a * np.exp(1j * pomak)



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
