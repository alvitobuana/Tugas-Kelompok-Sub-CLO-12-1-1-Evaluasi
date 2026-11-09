"""Mengecek Item-Based CF dengan contoh di slide M03.

Jalankan:  python tests/test_ibcf.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ibcf import ItemBasedCF, item_similarity  # noqa: E402

nan = np.nan
# Contoh film (slide M03). Baris: Amy, Calvin, Robert, David, Bradley
# Kolom: Pulp Fiction, Forrest Gump, The GodFather
FILM = np.array([
    [4, nan, 5],
    [5, 2, nan],
    [3, 3, 1],
    [nan, 2, 2],
    [1, 3, nan],
])

# Contoh Alice (slide M02/M03). Baris: Alice, User1..User4 ; Kolom: Item1..Item5
ALICE = np.array([
    [5, 3, 4, 4, nan],
    [3, 1, 2, 3, 3],
    [4, 3, 4, 3, 5],
    [3, 3, 1, 5, 4],
    [1, 5, 5, 2, 1],
])


def test_cosine_film_example():
    sim, _ = item_similarity(FILM, adjusted=False)
    assert np.isclose(sim[0, 1], 0.792, atol=1e-3)   # Pulp Fiction vs Forrest Gump
    assert np.isclose(sim[0, 2], 0.902, atol=1e-3)   # Pulp Fiction vs The GodFather


def test_prediction_formula_film_example():
    # Slide: prediksi Amy untuk Forrest Gump = (4*0.792 + 5*0.8) / (0.792 + 0.8) = 4.5
    model = ItemBasedCF(k=2, adjusted=False).fit(FILM)
    model.sim[1, 0] = model.sim[0, 1] = 0.792        # pakai similarity persis seperti slide
    model.sim[1, 2] = model.sim[2, 1] = 0.8
    assert np.isclose(model.predict(0, 1), 4.5, atol=0.01)


def test_cosine_and_adjusted_cosine_alice():
    cos, _ = item_similarity(ALICE, adjusted=False)
    adj, n = item_similarity(ALICE, adjusted=True)
    assert np.isclose(cos[4, 0], 0.99, atol=0.01)    # sim(I5, I1) cosine biasa
    assert np.isclose(adj[4, 0], 0.80, atol=0.01)    # sim(I5, I1) adjusted cosine
    assert n[4, 0] == 4                              # User1..User4 merating keduanya


def test_fallback_when_no_similar_item():
    R = np.array([[4, nan], [nan, 2.0]])
    model = ItemBasedCF().fit(R)
    assert model.predict(0, 1) == 4.0


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("LULUS", t.__name__)
    print(f"Semua {len(tests)} test lulus.")
