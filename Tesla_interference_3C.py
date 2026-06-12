"""
SRBIN Nikola Tesla, za sva vremena, najveci naucnik sveta.

SERBIAN Nikola Tesla, for all time, the greatest scientist in the world.
"""



"""
Tesla_interference_3C.py  —  GRUPA 3, deo C (interferencija + glavni program)

biranje kombinacija preko „interferencije" talasa iz frekvencije, 
zazora, parova i faznog talasa u talasnom polju

bira kombinacije iz talasnog polja po verovatnoci (verovatnoca = skor / max skor),
polje dolazi iz Tesla_quantum_wavefield_3B.

Ovo je jedini POKRETACKI fajl grupe 3 (3A i 3B su moduli). Daje txt + png + jpg.
"""


import numpy as np

from Tesla_Scalar_1 import (
    SEED,
    CSV_PATH,
    MIN_BROJ,
    MAX_BROJ,
    OUTPUT_DIR,
    ucitaj_izvlacenja,
)
from Tesla_quantum_wavefield_3B import (
    napravi_talasno_polje,
    normalizuj,
    TEZINA_FREKVENCIJE,
    TEZINA_ZAZORA,
    TEZINA_PAROVA,
    TEZINA_TALASA,
)

OSNOVA = "tesla_interference_3C"


def izaberi_interferencijom(tabela, broj_kombinacija=10, seed=SEED):
    """Izaberi 7/39 kombinacije bez ponavljanja brojeva, na osnovu talasnog polja."""
    rng = np.random.default_rng(seed)
    brojevi = tabela["broj"].to_numpy(dtype=int)
    p = normalizuj(tabela["verovatnoca"].to_numpy(dtype=float))

    kombinacije = []
    vidjeno = set()
    pokusaji = 0
    while len(kombinacije) < broj_kombinacija and pokusaji < broj_kombinacija * 200:
        pokusaji += 1
        izbor = tuple(sorted(rng.choice(brojevi, size=7, replace=False, p=p).tolist()))
        if izbor in vidjeno:
            continue
        vidjeno.add(izbor)
        kombinacije.append(izbor)
    return kombinacije


def skor_kombinacije(kombinacija, tabela):
    """Saberi skor pojedinacnih brojeva u jednoj kombinaciji."""
    mapa = dict(zip(tabela["broj"], tabela["skor"]))
    return float(sum(mapa[int(b)] for b in kombinacija))


def nacrtaj_skor(tabela, osnova=OSNOVA):
    """Nacrtaj skorove po brojevima (bar) i komponente, snimi PNG i JPG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = tabela.sort_values("broj")
    brojevi = t["broj"].to_numpy()
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle("Tesla GRUPA 3 - interferencija (frekvencija + zazor + parovi + fazni talas)")

    ax[0].bar(brojevi, t["skor"].to_numpy(), color="#1f77b4")
    ax[0].set_ylabel("skor")
    ax[0].set_xticks(brojevi)
    ax[0].grid(True, axis="y", alpha=0.3)

    ax[1].plot(brojevi, t["frekvencija_skor"].to_numpy(), label="frekvencija", marker="o", ms=3)
    ax[1].plot(brojevi, t["zazor_skor"].to_numpy(), label="zazor", marker="o", ms=3)
    ax[1].plot(brojevi, t["par_skor"].to_numpy(), label="parovi", marker="o", ms=3)
    ax[1].plot(brojevi, t["talas_skor"].to_numpy(), label="fazni talas", marker="o", ms=3)
    ax[1].set_xlabel("broj")
    ax[1].set_ylabel("komponente (0..1)")
    ax[1].set_xticks(brojevi)
    ax[1].legend(loc="upper right", ncol=4, fontsize=8)
    ax[1].grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    png = OUTPUT_DIR / f"{osnova}.png"
    jpg = OUTPUT_DIR / f"{osnova}.jpg"
    fig.savefig(png, dpi=150)
    fig.savefig(jpg, dpi=150)
    plt.close(fig)
    return png, jpg


def main():
    izvlacenja = ucitaj_izvlacenja()
    n = len(izvlacenja)
    tabela = napravi_talasno_polje(izvlacenja)
    kombinacije = izaberi_interferencijom(tabela, broj_kombinacija=10, seed=SEED)
    rangirane_kombinacije = sorted(
        ((k, skor_kombinacije(k, tabela)) for k in kombinacije),
        key=lambda kv: kv[1],
        reverse=True,
    )
    png, jpg = nacrtaj_skor(tabela, osnova=OSNOVA)

    with open(OUTPUT_DIR / f"{OSNOVA}.txt", "w", encoding="utf-8") as f:
        f.write("Tesla Scalar - GRUPA 3 / 3C (interferencija: frekvencija + zazor + parovi + fazni talas)\n")
        f.write(f"CSV: {CSV_PATH}\n")
        f.write(f"Izvlacenja: {n} | Seed: {SEED}\n")
        f.write(f"tezine: freq={TEZINA_FREKVENCIJE} zazor={TEZINA_ZAZORA} parovi={TEZINA_PAROVA} talas={TEZINA_TALASA}\n\n")
        f.write("Brojevi po skoru (talasno polje verovatnoce):\n")
        for _, r in tabela.iterrows():
            f.write(
                f"  {int(r['broj']):02d}  skor={r['skor']:.8f}  "
                f"freq={r['frekvencija_skor']:.4f}  zazor={r['zazor_skor']:.4f}  "
                f"par={r['par_skor']:.4f}  talas={r['talas_skor']:.4f}  "
                f"verovatnoca={r['verovatnoca']:.6f}\n"
            )

        f.write("\nPredlozene kombinacije (rangirane po skoru kombinacije):\n")
        for i, (k, s_komb) in enumerate(rangirane_kombinacije, start=1):
            f.write(f"  {i:02d}. " + " ".join(f"{v:02d}" for v in k) + f"  skor_komb={s_komb:.8f}\n")

        f.write("\nSlike polja/skora:\n")
        f.write(f"  PNG: {png}\n")
        f.write(f"  JPG: {jpg}\n")

    print()
    print("Tesla Scalar - GRUPA 3 / 3C (interferencija: frekvencija + zazor + parovi + fazni talas)")
    print(f"CSV: {CSV_PATH} | Izvlacenja: {n}")
    print(f"tezine: freq={TEZINA_FREKVENCIJE} zazor={TEZINA_ZAZORA} parovi={TEZINA_PAROVA} talas={TEZINA_TALASA}")
    print("\nTop 10 brojeva po skoru:")
    for _, r in tabela.head(10).iterrows():
        print(
            f"  {int(r['broj']):02d}  skor={r['skor']:.8f}  "
            f"freq={r['frekvencija_skor']:.4f}  zazor={r['zazor_skor']:.4f}  "
            f"par={r['par_skor']:.4f}  talas={r['talas_skor']:.4f}"
        )

    print("\nPredlozene kombinacije (rangirane po skoru kombinacije):")
    for i, (k, s_komb) in enumerate(rangirane_kombinacije, start=1):
        print(f"  {i:02d}. " + " ".join(f"{v:02d}" for v in k) + f"  skor_komb={s_komb:.8f}")
    print(f"\nSacuvano: {OUTPUT_DIR / f'{OSNOVA}.txt'}")
    print()


if __name__ == "__main__":
    main()



"""
Tesla Scalar - GRUPA 3 / 3C (interferencija: frekvencija + zazor + parovi + fazni talas)
CSV: /data/loto7hh_4630_k46.csv| Izvlacenja: 4630
tezine: freq=0.35 zazor=0.25 parovi=0.2 talas=0.2

Top 10 brojeva po skoru:
  39  skor=0.76260020  freq=0.5764  zazor=0.7368  par=0.8833  talas=1.0000
  21  skor=0.64350500  freq=0.4167  zazor=1.0000  par=0.4129  talas=0.8255
  08  skor=0.60203421  freq=1.0000  zazor=0.1579  par=0.2981  talas=0.7647
  23  skor=0.60053367  freq=0.9653  zazor=0.1053  par=0.4194  talas=0.7625
  35  skor=0.59969157  freq=0.5347  zazor=0.7895  par=0.3146  talas=0.7613
  32  skor=0.54955121  freq=0.6319  zazor=0.2632  par=0.6251  talas=0.6878
  29  skor=0.54096527  freq=0.5694  zazor=0.1579  par=0.8357  talas=0.6752
  05  skor=0.52494147  freq=0.4306  zazor=0.3158  par=0.8248  talas=0.6517
  34  skor=0.52179813  freq=0.7431  zazor=0.0000  par=0.6615  talas=0.6471
  33  skor=0.50974813  freq=0.6111  zazor=0.3684  par=0.3893  talas=0.6294

Predlozene kombinacije (rangirane po skoru kombinacije):
  01. 02 09 28 32 33 35 38  skor_komb=3.43822295
  02. 03 05 06 07 23 26 32  skor_komb=3.40826456
  03. 08 09 18 23 33 34 37  skor_komb=3.38499033
  04. 05 15 16 26 32 34 39  skor_komb=3.34548158
  05. 07 09 12 15 18 32 39  skor_komb=3.17816088
  06. 03 04 05 08 16 25 29  skor_komb=3.12951834
  07. 05 07 10 15 23 24 32  skor_komb=3.10498313
  08. 07 08 13 21 24 30 37  skor_komb=3.04027312
  09. 06 11 12 20 31 37 38  skor_komb=2.78450950
  10. 07 09 12 14 15 32 37  skor_komb=2.74018023

Sacuvano: /Tesla/tesla_interference_3C.txt
"""



"""
Imam težine. 
Pišem tri Tesla 3 fajla: 
3A (transform helperi), 3B (talasno polje), 3C (interferencija + main koji daje txt/png/jpg).

Tesla_wave_transform_3A.py — helperi (oblikuj_amplitudu, fazni_talas). Modul.
Tesla_quantum_wavefield_3B.py — talasno polje verovatnoće iz frekvencije + zazora + parova + faznog talasa (težine iz starog config.py: 0.35/0.25/0.20/0.20). Modul. 
Tesla_interference_3C.py — jedini pokretački fajl: učitava CSV, gradi polje preko 3B, bira kombinacije interferencijom, rangira po skor_komb, i daje tesla_interference_3C.txt/.png/.jpg.
Razlika u odnosu na grupu 1/2: ovde talas nije iz fizičkog solvera nego iz statistike izvlačenja + fazna modulacija. 
Zato sam stavio i drugačiji crtež (bar skorova + komponente freq/zazor/parovi/talas).
"""



"""
Kratka analiza — Tesla 3 (interferencija: frekvencija + zazor + parovi + fazni talas):

08 i 23 su na vrhu jer su jaki po svemu (freq=1.0/0.97, par=1.0/0.97) — to su frekvencijski lideri.
Novina je zazor: 21 (#5) ima zazor=1.0 (najduže nije izašao), 35/39 visok zazor → model nagrađuje „zakasnele" brojeve. To je drugačiji signal od grupe 1/2.
Jedno bitno zapažanje (moguća redundansa): 
kolone par i freq su skoro identične (08: oba 1.0; 23: oba 0.9653…). 
To je zato što je broj parova po broju ~ proporcionalan njegovoj frekvenciji (svako pojavljivanje daje 6 parova). 
Znači parovi ovde ne donose mnogo novog — praktično dva puta brojim frekvenciju.

par-skor da meri „jačinu najjačih parova" ili lift, a ne prost zbir 

Favorit 3C: 08 x 12 y 26 z 37 (skor_komb = 3.7383).

Sad imam kompletno: grupa 1, 2A-2D (fizički motori) i grupa 3 (statističko-talasni). 

_parovi u Tesla_quantum_wavefield_3B.py koriste lift (jačinu najjačih parova), ne prost zbir — time se uklanja dupliranje frekvencije 
Umesto prostog zbira parova (koji je bio ~ frekvencija), sada se računa lift: lift(a,b) = P(a,b) / (P(a)·P(b)) — koliko par izlazi zajedno više nego po slučaju.
Za svaki broj uzima se prosek top_k (6) najjačih lift-ova → „jačina najjačih parova".
Sad par kolona više neće biti kopija freq, nego nezavisan signal (parovi koji se vole zajedno). 

Primeri:
08: freq=1.0000, ali par=0.2981 → visok zbog frekvencije, ne zbog parova.
39: freq=0.5764, ali par=0.8833 i talas=1.0000 → zato je skočio na #1.
29: freq=0.5694, ali par=0.8357 → dobar par-signal.
05: freq=0.4306, ali par=0.8248 → odličan dokaz da lift radi.
21: zazor=1.0000 → model ga diže zbog čekanja, ne zbog frekvencije.
Ovo je sad mnogo bolje: freq, zazor, par i talas daju različite informacije.

Favorit 3C:
02 x 28 y 33 z 38 (skor_komb=3.4382)

3C je pravi nezavisan statističko-talasni model, ne samo frekvencijski model sa maskom.
"""



"""
Analiza — Tesla 3C (interferencija: frekvencija + zazor + parovi + fazni talas)

Priroda modela: za razliku od grupe 1 i 2 (fizički talasni motori), 3C je statističko-talasni. 
Skor svakog broja gradi se iz četiri komponente, sve skalirane na 0-1, pa ponderisane (freq 0.35 · zazor 0.25 · parovi 0.20 · talas 0.20), uz finalnu sinusnu faznu modulaciju.

Četiri signala (sada nezavisna):
frekvencija — koliko se broj pojavljuje (najveći: 08=1.0, 23=0.97).
zazor — koliko dugo nije izašao (najveći: 21=1.0, 35=0.79, 39=0.74).
parovi (lift) — sa kojim brojevima izlazi zajedno više nego po slučaju (najveći: 39=0.88, 29=0.84, 05=0.82).
fazni talas — sinusna modulacija preko baze (oblikuje vrhove).
Top 10 po skoru: 39 · x · 08 · y · 35 · z · 29 · x · 34 · 33

39 je #1 iako mu je frekvencija osrednja (0.58) — diže ga kombinacija zazora + parova + talasa. 
To je tačno ono što smo hteli: ne čista frekvencija.
21 je #2 skoro isključivo zbog zazora (1.0) — „najzakasneliji" broj.
08 i 23 su tu zbog frekvencije, ali su pali sa vrha jer im je par-lift slab (0.30/0.42) — više se ne dupliraju freq i parovi.
05 i 29 su „skriveni" brojevi: slabija frekvencija, ali jaki parovi → model ih ističe, što čista frekvencija nikad ne bi.
Favorit kombinacija: 02 x 28 y 33 z 38 (skor_komb = 3.4382).

Grupa 1/2 = fizika talasa (prostorni potpis).
Grupa 3 = istorijska statistika (frekvencija/zazor/parovi) + fazni talas. Ova dva pristupa su komplementarna. 3C je posebno koristan za hvatanje „zakasnelih" i „par-jakih" brojeva (21, 39, 05, 29) koje fizički motori ne vide.
Mali rizik: zazor reaguje na poslednje izvlačenje, pa je 3C osetljiviji na svežinu podataka — kad dođe novo kolo, rang se može primetno pomeriti (za razliku od fizičkih motora koji su stabilni).
"""



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
