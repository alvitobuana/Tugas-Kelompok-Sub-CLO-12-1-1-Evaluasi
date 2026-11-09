"""Metrik evaluasi sesuai slide dosen (Evaluasi Part 1 dan Part 2).

- MAE, MSE, RMSE : akurasi prediksi rating
- NMSE           : MSE / (r_max - r_min)
- NDCG@l         : kualitas urutan rekomendasi, DCG@l / IDCG@l
"""
import numpy as np
import pandas as pd


def mae(actual, pred):
    """Rata-rata selisih absolut |rec(u,i) - r(u,i)|."""
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    return np.mean(np.abs(pred - actual))


def mse(actual, pred):
    """Rata-rata selisih kuadrat (rec(u,i) - r(u,i))^2."""
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    return np.mean((pred - actual) ** 2)


def rmse(actual, pred):
    """Akar dari MSE, satuannya kembali sama dengan rating."""
    return np.sqrt(mse(actual, pred))


def nmse(actual, pred, r_min=1, r_max=5):
    """MSE dibagi rentang rating (slide: NMSE = MSE / (rmax - rmin))."""
    return mse(actual, pred) / (r_max - r_min)


def dcg(rels):
    """DCG = sum rel_i / log2(i + 1), dengan i = posisi 1, 2, 3, ..."""
    rels = np.asarray(rels, float)
    positions = np.arange(1, len(rels) + 1)
    return np.sum(rels / np.log2(positions + 1))


def ndcg_list(rels_in_predicted_order, l=None):
    """NDCG untuk satu daftar. IDCG = DCG dari urutan relevansi ideal (menurun)."""
    rels = np.asarray(rels_in_predicted_order, float)
    ideal = np.sort(rels)[::-1]
    if l is not None:
        rels, ideal = rels[:l], ideal[:l]
    idcg = dcg(ideal)
    return np.nan if idcg == 0 else dcg(rels) / idcg


def ndcg_at_l(df, l=10, threshold=4):
    """Rata-rata NDCG@l seluruh user.

    df berisi kolom user, item, rating (aktual), pred.
    Untuk setiap user, item di test set diurutkan berdasarkan prediksi
    (terbesar dulu). Relevansi = 1 jika rating aktual >= threshold, selain itu 0
    (aturan Kuis 2 di slide). User yang tidak punya satu pun item relevan
    di test set dilewati, karena IDCG-nya 0 sehingga NDCG tidak terdefinisi.
    """
    scores = []
    for _, g in df.groupby("user"):
        g = g.sort_values(["pred", "item"], ascending=[False, True])
        rels = (g["rating"].to_numpy() >= threshold).astype(float)
        scores.append(ndcg_list(rels, l))
    scores = np.array(scores)
    return np.nanmean(scores)


def evaluate(df, l=10, threshold=4, r_min=1, r_max=5):
    """Menghitung kelima metrik sekaligus dari tabel prediksi."""
    a, p = df["rating"].to_numpy(), df["pred"].to_numpy()
    return {
        "MAE": mae(a, p),
        "MSE": mse(a, p),
        "RMSE": rmse(a, p),
        "NMSE": nmse(a, p, r_min, r_max),
        f"NDCG@{l}": ndcg_at_l(df, l, threshold),
    }
