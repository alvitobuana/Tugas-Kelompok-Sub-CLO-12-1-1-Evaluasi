"""Mengecek User-Based CF dengan contoh Alice di slide M02.

Jalankan:  python tests/test_ubcf.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ubcf import UserBasedCF, pearson_similarity  # noqa: E402

nan = np.nan
# Baris: Alice, User1..User4 ; Kolom: Item1..Item5 (Alice belum merating Item5)
ALICE = np.array([
    [5, 3, 4, 4, nan],
    [3, 1, 2, 3, 3],
    [4, 3, 4, 3, 5],
    [3, 3, 1, 5, 4],
    [1, 5, 5, 2, 1],
])


def test_pearson_alice():
    # Slide: sim(Alice,User1)=0.85, User2=0.70, User3=0.00, User4=-0.79
    sim, n = pearson_similarity(ALICE)
    assert np.allclose(sim[0, 1:], [0.85, 0.70, 0.00, -0.79], atol=0.01)
    assert n[0, 1] == 4                      # co-rated: Item1..Item4


def test_prediction_alice_item5():
    # Slide: dengan tetangga User1 dan User2, prediksi Alice untuk Item5 = 4.87
    #   4 + (0.85*(3-2.4) + 0.70*(5-3.8)) / (0.85+0.70) = 4.87
    model = UserBasedCF(k=2).fit(ALICE)
    assert np.isclose(model.predict(0, 4), 4.87, atol=0.01)


def test_fallback_to_user_mean():
    R = np.array([[4, nan], [nan, 2.0]])     # tidak ada tetangga sama sekali
    model = UserBasedCF(k=5).fit(R)
    assert model.predict(0, 1) == 4.0


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("LULUS", t.__name__)
    print(f"Semua {len(tests)} test lulus.")
