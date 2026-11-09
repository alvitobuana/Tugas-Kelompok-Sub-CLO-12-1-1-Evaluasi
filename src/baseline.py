"""Metode pembanding (baseline) yang sangat sederhana.

Gunanya sebagai patokan: metode rekomendasi baru dianggap berguna jika hasilnya
jelas lebih baik daripada tebakan sesederhana ini.
"""
import numpy as np


class _Baseline:
    def __init__(self, r_min=1, r_max=5):
        self.r_min, self.r_max = r_min, r_max

    def fit(self, R):
        rated = ~np.isnan(R)
        X = np.nan_to_num(R)
        self.global_mean = X.sum() / rated.sum()
        self.user_mean = X.sum(axis=1) / rated.sum(axis=1)   # setiap user punya >= 20 rating
        n_item = rated.sum(axis=0)
        # film tanpa rating latih memakai rata-rata global
        self.item_mean = np.where(n_item > 0, X.sum(axis=0) / np.maximum(n_item, 1), self.global_mean)
        return self

    def _scores(self, u, i):
        raise NotImplementedError

    def predict_df(self, test):
        out = test.copy()
        u = test["user"].to_numpy() - 1
        i = test["item"].to_numpy() - 1
        out["pred"] = np.clip(self._scores(u, i), self.r_min, self.r_max)
        return out


class GlobalMean(_Baseline):
    """Selalu menebak rata-rata seluruh rating di data latih."""
    def _scores(self, u, i):
        return np.full(len(u), self.global_mean)


class UserMean(_Baseline):
    """Menebak rata-rata rating milik user tersebut."""
    def _scores(self, u, i):
        return self.user_mean[u]


class ItemMean(_Baseline):
    """Menebak rata-rata rating film tersebut (film tanpa rating latih: rata-rata global)."""
    def _scores(self, u, i):
        return self.item_mean[i]


class RandomGuess(_Baseline):
    """Menebak angka acak seragam antara 1 dan 5."""
    def __init__(self, seed=42, **kw):
        super().__init__(**kw)
        self.seed = seed

    def _scores(self, u, i):
        return np.random.default_rng(self.seed).uniform(self.r_min, self.r_max, len(u))
