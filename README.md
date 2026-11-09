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

Dataset **tidak disertakan** di repository ini karena lisensi MovieLens melarang penyebaran ulang
("The user may not redistribute the data without separate permission"). Dataset **diunduh otomatis** dari
situs resmi GroupLens saat notebook atau kode pertama kali dijalankan, lalu diekstrak ke `data/ml-100k/`.
Isi file unduhan dicocokkan dengan SHA-256 file resmi, sehingga semua anggota memakai data yang sama persis.

Jika komputer tidak terhubung ke internet, unduh manual
[ml-100k.zip](https://files.grouplens.org/datasets/movielens/ml-100k.zip) dan ekstrak ke folder `data/`
sehingga terdapat file `data/ml-100k/u.data`.

Sesuai ketentuan lisensi, penggunaan dataset ini dalam tulisan wajib mencantumkan:

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context.
> *ACM Transactions on Interactive Intelligent Systems (TiiS)* 5, 4, Article 19 (December 2015), 19 pages.
> https://doi.org/10.1145/2827872

## Struktur Repository

```
src/
  data.py          membaca dataset, membuat matriks rating user x film
  metrics.py       MAE, MSE, RMSE, NMSE, NDCG@l
  ubcf.py          user-based collaborative filtering
  ibcf.py          item-based collaborative filtering
  mf.py            matrix factorization (SGD)
  baseline.py      metode pembanding sederhana (acak, rata-rata)
  plot_style.py    gaya grafik
notebooks/
  01_eda.ipynb                  eksplorasi data
  02_user_based_cf.ipynb        user-based CF, langkah demi langkah
  03_item_based_cf.ipynb        item-based CF, langkah demi langkah
  04_matrix_factorization.ipynb matrix factorization, langkah demi langkah
  05_perbandingan.ipynb         perbandingan akhir ketiga metode dan baseline
scripts/
  mf_tuning.py     pemilihan hyperparameter MF dengan data validasi
tests/             pengujian kode dengan contoh hitungan di slide
figures/           grafik untuk laporan
results/           tabel hasil evaluasi (CSV)
PROGRESS.md        catatan progres dan alasan setiap keputusan
MASALAH_DAN_PENYELESAIAN.md   kendala yang ditemui dan cara penyelesaiannya
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
.venv/Scripts/python tests/test_baseline.py
```

Notebook dapat dibuka dengan Jupyter (`.venv/Scripts/python -m jupyter notebook`) atau Jupyter dari
Anaconda/VS Code, lalu dijalankan dengan *Run All*. Setiap notebook bisa dijalankan sendiri-sendiri; dataset
akan diunduh otomatis pada notebook pertama yang dijalankan. Notebook 04 dan 05 butuh sekitar 4–5 menit
karena melatih matrix factorization di 5 fold.

Untuk sekadar membaca hasil, notebook tidak perlu dijalankan: semua notebook sudah berisi output dan grafik
dan bisa dibuka langsung di GitHub.

## Metrik Evaluasi

| Metrik | Rumus | Keterangan |
|---|---|---|
| MAE | rata-rata \|prediksi − aktual\| | rata-rata besar kesalahan prediksi |
| MSE | rata-rata (prediksi − aktual)² | memberi hukuman besar pada kesalahan besar |
| RMSE | √MSE | satuan sama dengan rating |
| NMSE | MSE / (r_max − r_min) | MSE dinormalisasi rentang rating (1–5) |
| NDCG@10 | DCG@10 / IDCG@10 | kualitas urutan rekomendasi; film relevan jika rating aktual ≥ 4 |

## Hasil (rata-rata 5-fold)

| Metode | MAE | MSE | RMSE | NMSE | NDCG@10 |
|---|---|---|---|---|---|
| *Baseline: tebak acak* | 1,3902 | 2,8901 | 1,6999 | 0,7225 | 0,6839 |
| *Baseline: rata-rata user* | 0,8362 | 1,0895 | 1,0437 | 0,2724 | 0,7366 |
| *Baseline: rata-rata film* | 0,8174 | 1,0498 | 1,0246 | 0,2624 | 0,8329 |
| User-based CF (Pearson, k = 30) | **0,7342** | 0,8901 | 0,9434 | 0,2225 | 0,8474 |
| Item-based CF (adjusted cosine, k = 30) | 0,7383 | 0,8949 | 0,9459 | 0,2237 | 0,8375 |
| Matrix factorization (SGD, K = 100, β = 0,2) | 0,7357 | **0,8554** | **0,9249** | **0,2138** | **0,8578** |

Ringkasan temuan:

- Ketiga metode menurunkan MAE sekitar 10% dan RMSE 7,7–9,7% dibanding baseline terkuat (rata-rata film).
- Matrix factorization terbaik untuk RMSE dan NDCG@10, menang di kelima fold (uji t berpasangan, p < 0,05).
- Untuk MAE, ketiga metode praktis setara (selisih ≤ 0,0041, tidak signifikan).
- Semua metode kesulitan memprediksi film yang jarang dirating (sparsity), terutama item-based CF.

Penjelasan lengkap setiap langkah, keputusan, dan analisis ada di notebook, [PROGRESS.md](PROGRESS.md), dan
[MASALAH_DAN_PENYELESAIAN.md](MASALAH_DAN_PENYELESAIAN.md).

## Referensi

- Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets: History and Context. *ACM Transactions on
  Interactive Intelligent Systems*, 5(4), Article 19. https://doi.org/10.1145/2827872
- Jannach, D., Zanker, M., Felfernig, A., & Friedrich, G. (2010). *Recommender Systems: An Introduction*.
  Cambridge University Press.
- Sarwar, B., Karypis, G., Konstan, J., & Riedl, J. (2001). Item-based collaborative filtering recommendation
  algorithms. *Proceedings of the 10th International Conference on World Wide Web*, 285–295.
- Ricci, F., Rokach, L., & Shapira, B. (2011). Introduction to Recommender Systems Handbook. Dalam
  *Recommender Systems Handbook*. Springer.
- Materi kuliah Sistem Rekomendasi: User-Based CF, Item-Based CF, Matrix Factorization, dan Evaluasi
  Sistem Rekomendasi (ZK Abdurahman Baizal, Agung Toto Wibowo).
