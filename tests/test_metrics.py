"""Mengecek fungsi metrik dengan contoh hitungan yang ada di slide dosen.

Jalankan:  python tests/test_metrics.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.metrics import dcg, mae, mse, ndcg_at_l, ndcg_list, nmse, rmse  # noqa: E402


def test_mae_slide_evaluasi_tambahan():
    # Slide "Evaluasi (tambahan)": 4 rating tersembunyi, MAE = 7.0 / 4 = 1.75
    actual = [3, 4, 4, 5]
    pred = [5.0, 3.5, 2.0, 2.5]
    assert np.isclose(mae(actual, pred), 1.75)


def test_mse_rmse_nmse():
    actual, pred = [5, 3], [3, 3]          # error 2 dan 0
    assert np.isclose(mse(actual, pred), (4 + 0) / 2)
    assert np.isclose(rmse(actual, pred), np.sqrt(2))
    assert np.isclose(nmse(actual, pred), 2 / (5 - 1))


def test_dcg_ndcg_slide_evaluasi_part2():
    # Contoh slide Evaluasi Part 2: rel = 3,2,3,0,1,2
    rels = [3, 2, 3, 0, 1, 2]
    assert sum(rels) == 11                          # CG = 11
    assert np.isclose(dcg(rels), 6.861, atol=1e-3)   # DCG6 = 6.861
    assert np.isclose(dcg([3, 3, 2, 2, 1, 0]), 7.141, atol=1e-3)  # IDCG6
    assert np.isclose(ndcg_list(rels), 0.961, atol=1e-3)          # nDCG6


def test_ndcg_perfect_and_worst_order():
    df = pd.DataFrame({
        "user": [1, 1, 1],
        "item": [10, 20, 30],
        "rating": [5, 2, 4],     # item 10 dan 30 relevan (>= 4)
        "pred": [4.9, 1.0, 4.5], # urutan prediksi sempurna
    })
    assert np.isclose(ndcg_at_l(df, l=3), 1.0)
    df["pred"] = [1.0, 5.0, 2.0]  # item tidak relevan malah di posisi 1
    assert ndcg_at_l(df, l=3) < 1.0


def test_user_without_relevant_item_is_skipped():
    df = pd.DataFrame({
        "user": [1, 1, 2, 2],
        "item": [1, 2, 1, 2],
        "rating": [5, 1, 2, 1],   # user 2 tidak punya item relevan
        "pred": [5, 1, 3, 2],
    })
    assert np.isclose(ndcg_at_l(df, l=2), 1.0)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("LULUS", t.__name__)
    print(f"Semua {len(tests)} test lulus.")
