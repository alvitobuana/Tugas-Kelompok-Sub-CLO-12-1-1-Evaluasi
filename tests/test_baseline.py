"""Mengecek baseline sederhana.

Jalankan:  python tests/test_baseline.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.baseline import GlobalMean, ItemMean, RandomGuess, UserMean  # noqa: E402

nan = np.nan
R = np.array([[5, 3, nan],
              [1, nan, nan]])          # film ke-3 tidak punya rating latih
TEST = pd.DataFrame({"user": [1, 2, 2], "item": [3, 2, 3], "rating": [4, 2, 3]})


def test_global_mean():
    p = GlobalMean().fit(R).predict_df(TEST)["pred"]
    assert np.allclose(p, (5 + 3 + 1) / 3)


def test_user_mean():
    p = UserMean().fit(R).predict_df(TEST)["pred"]
    assert np.allclose(p, [4.0, 1.0, 1.0])


def test_item_mean_with_fallback():
    p = ItemMean().fit(R).predict_df(TEST)["pred"]
    assert np.allclose(p, [3.0, 3.0, 3.0])   # film 2 = 3; film 3 = rata-rata global 3


def test_random_in_range_and_repeatable():
    a = RandomGuess(seed=1).fit(R).predict_df(TEST)["pred"]
    b = RandomGuess(seed=1).fit(R).predict_df(TEST)["pred"]
    assert np.all((a >= 1) & (a <= 5)) and np.allclose(a, b)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("LULUS", t.__name__)
    print(f"Semua {len(tests)} test lulus.")
