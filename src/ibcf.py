"""Item-Based Collaborative Filtering (slide M03, Sarwar et al. 2001).

Langkah:
1. Hitung kemiripan antar-film dari pola rating para user, memakai
   cosine similarity biasa atau adjusted cosine similarity.
2. Untuk memprediksi rating user u pada film p, ambil k film yang paling mirip
   dengan p dan SUDAH dirating oleh user u.
3. Prediksi = rata-rata tertimbang rating user u pada film-film tersebut,
   dengan bobot = similarity.
"""
import numpy as np


def item_similarity(R, adjusted=True):
    """Matriks similarity antar-film (kolom R) dan jumlah user co-rated.

    Cosine biasa (slide M03, contoh Pulp Fiction vs Forrest Gump):
        sim(a,b) = sum_u r_ua * r_ub / ( sqrt(sum_u r_ua^2) * sqrt(sum_u r_ub^2) )

    Adjusted cosine (slide M03, contoh Alice):
        sim(a,b) = sum_u (r_ua - mean_u)(r_ub - mean_u)
                   / ( sqrt(sum_u (r_ua - mean_u)^2) * sqrt(sum_u (r_ub - mean_u)^2) )

    Pada kedua rumus, u hanya user yang merating KEDUA film. mean_u adalah
    rata-rata seluruh rating milik user u. Adjusted cosine menghilangkan
    perbedaan kebiasaan user (ada yang pelit, ada yang murah hati memberi rating).
    """
    M = (~np.isnan(R)).astype(float)                 # user x item, 1 jika ada rating
    X = R - np.nanmean(R, axis=1, keepdims=True) if adjusted else R.copy()
    X = np.nan_to_num(X)                             # sel kosong = 0, tidak ikut dijumlah
    Xt, Mt, X2t = X.T, M.T, (X ** 2).T               # item x user

    n = Mt @ Mt.T                                    # jumlah user co-rated
    num = Xt @ Xt.T                                  # sum_u x_ua * x_ub (otomatis hanya co-rated)
    norm_a = X2t @ Mt.T                              # sum_u x_ua^2, hanya user yang juga merating b
    norm_b = norm_a.T
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = num / np.sqrt(norm_a * norm_b)
    sim[(n < 1) | ~np.isfinite(sim)] = 0.0
    np.fill_diagonal(sim, 0.0)                       # film tidak dihitung mirip dengan dirinya sendiri
    return np.clip(sim, -1, 1), n


class ItemBasedCF:
    """Item-based CF dengan k film tetangga.

    Parameter
    ---------
    k : jumlah film mirip yang dipakai untuk prediksi.
    adjusted : True = adjusted cosine, False = cosine biasa.
    sig_weight : jika diisi gamma, similarity dikalikan min(n, gamma)/gamma
        (significance weighting) agar pasangan film dengan sedikit user
        co-rated tidak dianggap sangat mirip.
    min_sim : threshold similarity (slide M03); hanya film dengan sim > min_sim dipakai.
    """

    def __init__(self, k=30, adjusted=True, sig_weight=None, min_sim=0.0, r_min=1, r_max=5):
        self.k = k
        self.adjusted = adjusted
        self.sig_weight = sig_weight
        self.min_sim = min_sim
        self.r_min, self.r_max = r_min, r_max

    def fit(self, R):
        self.R = R
        self.rated = ~np.isnan(R)
        self.user_mean = np.nanmean(R, axis=1)
        self.sim, self.n_common = item_similarity(R, self.adjusted)
        if self.sig_weight:
            g = self.sig_weight
            self.sim = self.sim * np.minimum(self.n_common, g) / g
        self.rated_items = [np.flatnonzero(self.rated[u]) for u in range(R.shape[0])]
        return self

    def neighbors(self, u, p):
        """k film paling mirip dengan film p (sim > min_sim) yang sudah dirating user u."""
        cand = self.rated_items[u]
        s = self.sim[p, cand]
        keep = s > self.min_sim
        cand, s = cand[keep], s[keep]
        if len(cand) > self.k:
            top = np.argpartition(-s, self.k)[: self.k]
            cand, s = cand[top], s[top]
        return cand, s

    def _predict(self, u, p):
        cand, s = self.neighbors(u, p)
        if len(cand) == 0:
            # Tidak ada film mirip yang pernah dirating user (misalnya film baru
            # yang tidak ada di data latih) -> cadangan: rata-rata rating user.
            return self.user_mean[u], 0
        pred = np.sum(s * self.R[u, cand]) / np.sum(s)
        return float(np.clip(pred, self.r_min, self.r_max)), len(cand)

    def predict(self, u, p):
        """Prediksi rating user u (indeks 0) untuk film p (indeks 0)."""
        return self._predict(u, p)[0]

    def predict_df(self, test):
        """Menambahkan kolom 'pred' dan 'n_tetangga' ke tabel test (id 1-based)."""
        out = test.copy()
        res = [self._predict(u - 1, i - 1)
               for u, i in zip(test["user"].to_numpy(), test["item"].to_numpy())]
        out["pred"] = [r[0] for r in res]
        out["n_tetangga"] = [r[1] for r in res]
        return out

    def recommend(self, u, n=10, min_neighbors=5):
        """Top-n film yang belum dirating user u, dengan minimal `min_neighbors` film pendukung."""
        scores = []
        for p in np.flatnonzero(~self.rated[u]):
            pred, cnt = self._predict(u, p)
            if cnt >= min_neighbors:
                scores.append((p, pred, cnt))
        scores.sort(key=lambda t: (-t[1], -t[2], t[0]))
        return scores[:n]
