"""Mengecek Matrix Factorization dengan contoh di slide M04.

Jalankan:  python tests/test_mf.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.mf import MatrixFactorization  # noqa: E402

nan = np.nan
# Matriks contoh slide M04 (U1..U5 x D1..D4), 0 di slide = belum ada rating
SLIDE_R = np.array([
    [5, 3, nan, 1],
    [4, nan, nan, 1],
    [1, 1, nan, 5],
    [1, nan, nan, 4],
    [nan, 1, 5, 4],
])


def test_slide_example_reconstructs_known_ratings():
    # Pengaturan persis seperti kode slide: K=2, alpha=0.0002, beta=0.02, 5000 iterasi
    m = MatrixFactorization(K=2, alpha=0.0002, beta=0.02, epochs=5000, init="slide", seed=0)
    m.fit(SLIDE_R)
    approx = m.P @ m.Q.T
    known = ~np.isnan(SLIDE_R)
    # Di output slide, rating yang diketahui terulang dengan selisih < 0.4
    assert np.max(np.abs(approx[known] - SLIDE_R[known])) < 0.4
    # Error latih harus turun jauh dibanding epoch pertama
    assert m.history[-1]["rmse_train"] < 0.3 < m.history[0]["rmse_train"]


def test_one_sgd_step_reduces_error():
    R = np.array([[4.0]])
    m = MatrixFactorization(K=2, alpha=0.05, beta=0.0, epochs=1, init="slide", seed=1)
    m._init_factors(1, 1, 4.0)
    e_before = 4.0 - m.P[0] @ m.Q[0]
    m.fit(R)
    e_after = 4.0 - m.P[0] @ m.Q[0]
    assert abs(e_after) < abs(e_before)


def test_unseen_item_falls_back_to_user_mean():
    R = np.array([[4, 2, nan], [5, 3, nan]])   # film ke-3 tidak punya rating latih
    m = MatrixFactorization(K=2, epochs=5).fit(R)
    assert m.predict(0, 2) == 3.0


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("LULUS", t.__name__)
    print(f"Semua {len(tests)} test lulus.")
