"""Memilih hyperparameter Matrix Factorization TANPA menyentuh data uji.

Data latih fold 1 (u1.base) dibagi lagi: 90% untuk melatih, 10% sebagai data
validasi. Setiap konfigurasi dilatih dan RMSE validasinya dicatat per epoch.
Hasil disimpan di results/mf_tuning.csv lalu dianalisis di notebook 04.

Jalankan (sekitar 10 menit):  python scripts/mf_tuning.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import load_fold, to_matrix  # noqa: E402
from src.mf import MatrixFactorization  # noqa: E402

EPOCHS = 60


def split_validation(train, frac=0.1, seed=42):
    rng = np.random.default_rng(seed)
    is_val = rng.random(len(train)) < frac
    return train[~is_val], train[is_val]


def main():
    train, _ = load_fold(1)
    sub, val = split_validation(train)
    R_sub, R_val = to_matrix(sub), to_matrix(val)
    print(f"latih {len(sub)} rating, validasi {len(val)} rating")

    configs = []
    # Tahap 1: pengaruh learning rate (termasuk nilai slide 0.0002)
    for alpha in [0.0002, 0.005, 0.01]:
        configs.append(dict(tahap="learning rate", K=20, alpha=alpha, beta=0.02))
    # Tahap 2: jumlah fitur K dan regularisasi beta
    for K in [5, 10, 20, 50]:
        for beta in [0.02, 0.1]:
            if not (K == 20 and beta == 0.02):
                configs.append(dict(tahap="K dan beta", K=K, alpha=0.01, beta=beta))
    configs.append(dict(tahap="K dan beta", K=20, alpha=0.01, beta=0.02))

    rows = []
    for c in configs:
        t = time.time()
        m = MatrixFactorization(K=c["K"], alpha=c["alpha"], beta=c["beta"], epochs=EPOCHS)
        m.fit(R_sub, R_val)
        for h in m.history:
            rows.append({**c, **h})
        best = min(m.history, key=lambda h: h["rmse_val"])
        print(f"{c} -> RMSE val terbaik {best['rmse_val']:.4f} di epoch {best['epoch']} "
              f"({time.time() - t:.0f} detik)")
        pd.DataFrame(rows).to_csv(ROOT / "results" / "mf_tuning.csv", index=False)


if __name__ == "__main__":
    main()
