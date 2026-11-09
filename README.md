# Prediksi Rating dan Rekomendasi Item dengan Collaborative Filtering dan Matrix Factorization

Tugas Kelompok Sistem Rekomendasi (Sub-CLO 12-1-1, Evaluasi), kelas RecSys 2627-1.
Dosen: ZK Abdurahman Baizal.

**Kelompok 3**

| No | Nama |
|----|------|
| 1 | Daffa Irsandy Putra |
| 2 | Alvito Buana |
| 3 | Zweta Lathifah Kus Aliyyah |
| 4 | Muhammad Ragil |
| 5 | Laili Faisa Dwi Cahyarini |

## Deskripsi

Proyek ini membangun sistem prediksi rating dan rekomendasi film dengan tiga metode:

1. **User-based collaborative filtering**: kemiripan antar-user dengan Pearson correlation.
2. **Item-based collaborative filtering**: kemiripan antar-film dengan cosine dan adjusted cosine similarity.
3. **Matrix factorization**: faktorisasi matriks rating dengan stochastic gradient descent.

Ketiga metode dievaluasi dengan metrik yang ada di slide perkuliahan: **MAE, MSE, RMSE, NMSE, dan NDCG**.
Semua metode dan metrik ditulis sendiri dengan NumPy/Pandas (tanpa library rekomendasi siap pakai), dan
setiap rumus diuji dengan contoh hitungan dari slide.

## Dataset

[MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) dari GroupLens Research: 100.000 rating
(skala 1–5) dari 943 user untuk 1.682 film, lengkap dengan timestamp. Setiap user memiliki minimal 20 rating.
Evaluasi memakai pembagian 5-fold bawaan dataset (`u1.base/u1.test` sampai `u5.base/u5.test`).

Dataset tidak disertakan di repository ini. Unduh dan ekstrak ke folder `data/`:

```bash
mkdir -p data
curl -L -o data/ml-100k.zip https://files.grouplens.org/datasets/movielens/ml-100k.zip
python -c "import zipfile; zipfile.ZipFile('data/ml-100k.zip').extractall('data')"
```

Setelah diekstrak, file rating berada di `data/ml-100k/u.data`.

## Struktur Repository

```
src/
  data.py          membaca dataset, membuat matriks rating user x film
  metrics.py       MAE, MSE, RMSE, NMSE, NDCG@l
  ubcf.py          user-based collaborative filtering
  ibcf.py          item-based collaborative filtering
  mf.py            matrix factorization (SGD)
  plot_style.py    gaya grafik
notebooks/
  01_eda.ipynb                  eksplorasi data
  02_user_based_cf.ipynb        user-based CF, langkah demi langkah
  03_item_based_cf.ipynb        item-based CF, langkah demi langkah
  04_matrix_factorization.ipynb matrix factorization, langkah demi langkah
scripts/
  mf_tuning.py     pemilihan hyperparameter MF dengan data validasi
tests/             pengujian kode dengan contoh hitungan di slide
figures/           grafik untuk laporan
results/           tabel hasil evaluasi (CSV)
PROGRESS.md        catatan progres dan alasan setiap keputusan
```

## Cara Menjalankan

Membutuhkan Python 3.11 atau lebih baru.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt      # Windows
# .venv/bin/python -m pip install -r requirements.txt        # Linux/macOS
```

Menjalankan pengujian (memastikan rumus sama dengan contoh di slide):

```bash
.venv/Scripts/python tests/test_metrics.py
.venv/Scripts/python tests/test_ubcf.py
.venv/Scripts/python tests/test_ibcf.py
.venv/Scripts/python tests/test_mf.py
```

Notebook dapat dibuka dengan Jupyter (`.venv/Scripts/python -m jupyter notebook`) dan dijalankan berurutan
dari 01. Semua notebook sudah berisi output hasil eksekusi.

## Metrik Evaluasi

| Metrik | Rumus | Keterangan |
|---|---|---|
| MAE | rata-rata \|prediksi − aktual\| | rata-rata besar kesalahan prediksi |
| MSE | rata-rata (prediksi − aktual)² | memberi hukuman besar pada kesalahan besar |
| RMSE | √MSE | satuan sama dengan rating |
| NMSE | MSE / (r_max − r_min) | MSE dinormalisasi rentang rating (1–5) |
| NDCG@10 | DCG@10 / IDCG@10 | kualitas urutan rekomendasi; film relevan jika rating aktual ≥ 4 |

## Hasil Sementara (rata-rata 5-fold)

| Metode | MAE | MSE | RMSE | NMSE | NDCG@10 |
|---|---|---|---|---|---|
| User-based CF (Pearson, k = 30) | 0,7342 | 0,8901 | 0,9434 | 0,2225 | 0,8474 |
| Item-based CF (adjusted cosine, k = 30) | 0,7383 | 0,8949 | 0,9459 | 0,2237 | 0,8375 |
| Matrix factorization (SGD, K = 100, β = 0,2) | 0,7357 | 0,8554 | 0,9249 | 0,2138 | 0,8578 |

Penjelasan lengkap setiap langkah, keputusan, dan analisis ada di notebook dan di [PROGRESS.md](PROGRESS.md).
