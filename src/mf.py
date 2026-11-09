"""Matrix Factorization dengan Stochastic Gradient Descent (slide M04).

Ide: matriks rating R (user x film) didekati dengan perkalian dua matriks kecil

    R  ~  P x Q^T

- P (user x K): setiap baris = "selera" seorang user terhadap K fitur tersembunyi
- Q (film x K): setiap baris = seberapa kuat sebuah film memiliki K fitur tersebut
- Prediksi rating user i untuk film j = p_i . q_j (dot product)

Untuk setiap rating yang diketahui, error e_ij = r_ij - p_i . q_j dihitung lalu
P dan Q digeser sedikit ke arah yang mengecilkan error (rumus update slide):

    p_ik <- p_ik + alpha * (2 * e_ij * q_kj - beta * p_ik)
    q_kj <- q_kj + alpha * (2 * e_ij * p_ik - beta * q_kj)

alpha = learning rate (besar langkah), beta = regularisasi (mencegah nilai P dan Q
membesar berlebihan sehingga model hanya "menghafal" data latih).
"""
import numpy as np


class MatrixFactorization:
    """Matrix factorization tanpa bias, dilatih dengan SGD seperti slide M04.

    Parameter
    ---------
    K : jumlah fitur tersembunyi (latent features).
    alpha : learning rate.
    beta : koefisien regularisasi.
    epochs : berapa kali seluruh rating latih dilalui.
    init : "slide" = acak seragam 0..1 seperti kode slide;
           "scaled" = acak seragam 0..s dengan s dipilih agar rata-rata
           prediksi awal (p . q) kira-kira sama dengan rata-rata rating.
    seed : agar hasil acak bisa diulang.
    """

    def __init__(self, K=20, alpha=0.01, beta=0.02, epochs=50, init="scaled",
                 seed=42, r_min=1, r_max=5):
        self.K, self.alpha, self.beta, self.epochs = K, alpha, beta, epochs
        self.init, self.seed = init, seed
        self.r_min, self.r_max = r_min, r_max

    def _init_factors(self, n_users, n_items, mu):
        rng = np.random.default_rng(self.seed)
        if self.init == "slide":
            s = 1.0
        else:
            # E[p . q] = K * (s/2)^2 = mu  ->  s = 2 * sqrt(mu / K)
            s = 2 * np.sqrt(mu / self.K)
        self.P = rng.uniform(0, s, (n_users, self.K))
        self.Q = rng.uniform(0, s, (n_items, self.K))
        return rng

    def fit(self, R, R_val=None, verbose=False):
        """Melatih P dan Q dari matriks R (NaN = tidak ada rating).

        Jika R_val diberikan (rating validasi, tidak dipakai untuk melatih),
        RMSE validasi dicatat di setiap epoch untuk memilih jumlah epoch.
        """
        n_users, n_items = R.shape
        users, items = np.nonzero(~np.isnan(R))
        ratings = R[users, items]
        self.user_mean = np.nanmean(R, axis=1)
        self.item_seen = ~np.all(np.isnan(R), axis=0)      # film yang punya rating latih
        self.item_count = np.sum(~np.isnan(R), axis=0)
        rng = self._init_factors(n_users, n_items, ratings.mean())

        self.history = []
        P, Q, a, b = self.P, self.Q, self.alpha, self.beta
        for epoch in range(1, self.epochs + 1):
            # "Stochastic": urutan rating diacak di setiap epoch
            for idx in rng.permutation(len(ratings)):
                i, j = users[idx], items[idx]
                p, q = P[i].copy(), Q[j].copy()            # nilai lama, agar P dan Q diupdate serentak
                e = ratings[idx] - p @ q
                P[i] += a * (2 * e * q - b * p)
                Q[j] += a * (2 * e * p - b * q)
            rec = {"epoch": epoch, "rmse_train": self._rmse(users, items, ratings)}
            if R_val is not None:
                vu, vi = np.nonzero(~np.isnan(R_val))
                rec["rmse_val"] = self._rmse(vu, vi, R_val[vu, vi])
            self.history.append(rec)
            if verbose:
                print(rec)
        return self

    def _rmse(self, users, items, ratings):
        pred = np.clip(np.sum(self.P[users] * self.Q[items], axis=1), self.r_min, self.r_max)
        return float(np.sqrt(np.mean((ratings - pred) ** 2)))

    def predict(self, u, i):
        """Prediksi rating user u untuk film i (indeks 0).

        Film yang tidak punya satu pun rating di data latih tidak pernah
        diperbarui (vektor q-nya masih acak), jadi prediksinya diganti
        rata-rata rating user, sama seperti cadangan pada metode CF lain.
        """
        if not self.item_seen[i]:
            return float(self.user_mean[u])
        return float(np.clip(self.P[u] @ self.Q[i], self.r_min, self.r_max))

    def predict_df(self, test):
        """Menambahkan kolom 'pred' ke tabel test (id 1-based)."""
        out = test.copy()
        u = test["user"].to_numpy() - 1
        i = test["item"].to_numpy() - 1
        pred = np.clip(np.sum(self.P[u] * self.Q[i], axis=1), self.r_min, self.r_max)
        out["pred"] = np.where(self.item_seen[i], pred, self.user_mean[u])
        return out

    def recommend(self, u, rated, n=10, min_ratings=5):
        """Top-n film yang belum dirating user u.

        Film yang dirating kurang dari `min_ratings` user di data latih
        dilewati, karena vektor q-nya dilatih dari terlalu sedikit data.
        """
        scores = np.clip(self.Q @ self.P[u], self.r_min, self.r_max)
        ok = (~rated) & (self.item_count >= min_ratings)
        cand = np.flatnonzero(ok)
        order = cand[np.lexsort((cand, -scores[cand]))]
        return [(i, float(scores[i]), int(self.item_count[i])) for i in order[:n]]
