"""User-Based Collaborative Filtering (slide M02, Jannach et al. 2010).

Langkah:
1. Hitung kemiripan antar-user dengan Pearson correlation, hanya pada item
   yang dirating oleh kedua user (co-rated items).
2. Untuk memprediksi rating user a pada item i, ambil k tetangga paling mirip
   (similarity positif) yang sudah merating item i.
3. Prediksi = rata-rata rating user a
             + rata-rata tertimbang dari (rating tetangga - rata-rata tetangga).
"""
import numpy as np


def pearson_similarity(R):
    """Matriks Pearson correlation antar-user (baris) dari matriks R berisi NaN.

    Untuk pasangan user (a, b), korelasi dihitung hanya pada item P yang dirating
    keduanya, dengan rata-rata masing-masing user dihitung di P:

        sim(a,b) = sum_p (r_ap - mean_a)(r_bp - mean_b)
                   / ( sqrt(sum_p (r_ap - mean_a)^2) * sqrt(sum_p (r_bp - mean_b)^2) )

    Supaya cepat, rumus di atas ditulis ulang dalam bentuk perkalian matriks
    (cov = sum xy - sum x * sum y / n), sehingga semua pasangan dihitung sekaligus.
    Juga mengembalikan n_common: jumlah item co-rated untuk setiap pasangan.
    """
    M = (~np.isnan(R)).astype(float)      # 1 jika ada rating, 0 jika kosong
    X = np.nan_to_num(R)                  # rating, kosong diisi 0
    X2 = X ** 2

    n = M @ M.T                           # jumlah item co-rated
    Sa = X @ M.T                          # jumlah rating a di item co-rated
    Sb = Sa.T                             # jumlah rating b di item co-rated
    Saa = X2 @ M.T
    Sbb = Saa.T
    Sab = X @ X.T

    with np.errstate(divide="ignore", invalid="ignore"):
        cov = Sab - Sa * Sb / n
        var_a = Saa - Sa ** 2 / n
        var_b = Sbb - Sb ** 2 / n
        sim = cov / np.sqrt(var_a * var_b)

    # Korelasi tidak terdefinisi jika item co-rated < 2 atau salah satu user
    # memberi rating yang sama semua (variansi 0) -> anggap tidak mirip (0).
    sim[(n < 2) | ~np.isfinite(sim)] = 0.0
    np.fill_diagonal(sim, 0.0)            # user tidak dihitung sebagai tetangganya sendiri
    return np.clip(sim, -1, 1), n


class UserBasedCF:
    """User-based CF dengan Pearson correlation dan k tetangga terdekat.

    Parameter
    ---------
    k : jumlah tetangga yang dipakai untuk prediksi (slide: 20-50 wajar).
    sig_weight : jika diisi angka gamma, similarity dikalikan min(n, gamma)/gamma
        ("significance weighting" di slide M02) sehingga pasangan user yang
        hanya punya sedikit item co-rated tidak dianggap sangat mirip.
    r_min, r_max : batas rating, prediksi dipotong ke rentang ini.
    """

    def __init__(self, k=30, sig_weight=None, r_min=1, r_max=5):
        self.k = k
        self.sig_weight = sig_weight
        self.r_min, self.r_max = r_min, r_max

    def fit(self, R):
        self.R = R
        self.rated = ~np.isnan(R)
        self.user_mean = np.nanmean(R, axis=1)          # rata-rata rating tiap user
        self.dev = R - self.user_mean[:, None]           # r_bi - mean_b
        self.sim, self.n_common = pearson_similarity(R)
        if self.sig_weight:
            g = self.sig_weight
            self.sim = self.sim * np.minimum(self.n_common, g) / g
        # daftar user yang pernah merating tiap item (agar pencarian tetangga cepat)
        self.raters = [np.flatnonzero(self.rated[:, i]) for i in range(R.shape[1])]
        return self

    def neighbors(self, a, i):
        """k tetangga user a (similarity > 0) yang sudah merating item i."""
        cand = self.raters[i]
        s = self.sim[a, cand]
        keep = s > 0
        cand, s = cand[keep], s[keep]
        if len(cand) > self.k:
            top = np.argpartition(-s, self.k)[: self.k]
            cand, s = cand[top], s[top]
        return cand, s

    def predict(self, a, i):
        """Prediksi rating user a (indeks 0) untuk item i (indeks 0)."""
        cand, s = self.neighbors(a, i)
        if len(cand) == 0:
            pred = self.user_mean[a]     # fallback: rata-rata user (seperti slide tambahan)
        else:
            pred = self.user_mean[a] + np.sum(s * self.dev[cand, i]) / np.sum(s)
        return float(np.clip(pred, self.r_min, self.r_max))

    def predict_df(self, test):
        """Menambahkan kolom 'pred' ke tabel test (kolom user, item 1-based)."""
        out = test.copy()
        out["pred"] = [
            self.predict(u - 1, i - 1)
            for u, i in zip(test["user"].to_numpy(), test["item"].to_numpy())
        ]
        return out

    def recommend(self, a, n=10, min_neighbors=5):
        """Top-n item yang belum dirating user a, diurutkan dari prediksi tertinggi.

        Item yang dirating kurang dari `min_neighbors` tetangga dilewati.
        Tanpa syarat ini, film obscure yang kebetulan diberi rating 5 oleh
        satu-dua tetangga akan selalu menempati peringkat teratas, padahal
        buktinya sangat lemah.
        """
        scores = []
        for i in np.flatnonzero(~self.rated[a]):
            cand, s = self.neighbors(a, i)
            if len(cand) >= min_neighbors:
                p = self.user_mean[a] + np.sum(s * self.dev[cand, i]) / np.sum(s)
                scores.append((i, float(np.clip(p, self.r_min, self.r_max)), len(cand)))
        scores.sort(key=lambda t: (-t[1], -t[2], t[0]))
        return scores[:n]
