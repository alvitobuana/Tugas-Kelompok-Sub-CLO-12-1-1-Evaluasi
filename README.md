# Prediksi Rating dan Rekomendasi Item dengan Collaborative Filtering dan Matrix Factorization

Tugas Kelompok Sistem Rekomendasi (Sub-CLO 12-1-1, Evaluasi), kelas RecSys 2627-1.  
Program Studi S1 Informatika, Universitas Telkom.  
Dosen Pengampu: Prof. Dr. Z.K. Abdurahman Baizal, S.Si., M.Kom.

**Kelompok 3**

| No | Nama | NIM |
|:---|:---|:---|
| 1 | Daffa Irsandy Putra | 103012300338 |
| 2 | Alvito Buana | 103012300232 |
| 3 | Zweta Lathifah Kus Aliyyah | 103012300107 |
| 4 | Muhammad Ragil | 103012300015 |
| 5 | Laili Faisa Dwi Cahyarini | 103012300431 |

---

## Ringkasan Proyek

Proyek ini membangun dan mengevaluasi sistem prediksi rating dan rekomendasi film dengan tiga metode utama:

1. **User-Based Collaborative Filtering (UBCF)**: Mengukur kemiripan preferensi antar-pengguna dengan *Pearson correlation* dan *significance weighting* ($\gamma = 50$).
2. **Item-Based Collaborative Filtering (IBCF)**: Mengukur kemiripan pola penilaian antar-film dengan *Adjusted Cosine Similarity* dan *significance weighting*.
3. **Matrix Factorization (MF)**: Memproyeksikan pengguna dan item ke ruang fitur laten bersama ($K = 100$) menggunakan optimasi *Stochastic Gradient Descent* (SGD) dengan regularisasi $L_2$ ($\beta = 0{,}2$).

Ketiga algoritma dievaluasi terhadap empat model pembanding (*baseline*: tebakan acak, rata-rata global, rata-rata pengguna, dan rata-rata item) menggunakan metrik standar pada materi perkuliahan: **MAE, MSE, RMSE, NMSE, dan NDCG@10**. Seluruh metode dan metrik ditulis sendiri dari rumus dasar menggunakan NumPy dan Pandas tanpa dependensi pustaka rekomendasi *black-box*.

---

## Dataset

[MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) dari GroupLens Research:
- **100.000 rating** (skala 1–5 bulat) dari 943 pengguna untuk 1.682 film, disertai timestamp.
- Setiap pengguna memiliki minimal 20 riwayat penilaian (*user history*).
- Evaluasi menggunakan protokol **5-fold cross-validation** bawaan resmi (`u1.base/u1.test` hingga `u5.base/u5.test`), masing-masing 80.000 data latih dan 20.000 data uji tanpa irisan.

Dataset **diunduh otomatis** dari peladen resmi GroupLens saat kode pertama kali dijalankan dan diverifikasi terhadap nilai SHA-256 berkas resmi (`50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229`).

---

## Struktur Repositori

```
├── notebooks/
│   └── Evaluasi_RecSys_CF_MF.ipynb                   ← NOTEBOOK UTAMA: Lembar kerja teknis praktis & verifikasi
├── laporan/                                          ← FOLDER LAPORAN AKADEMIK LATEX (Resmi FIF Telkom University)
│   ├── main.tex                                      ← Induk berkas LaTeX (tanpa Lembar Pengesahan)
│   ├── main.pdf                                      ← Dokumen PDF laporan akademik hasil kompilasi LaTeX (41 halaman)
│   ├── Cover.tex                                     ← Halaman sampul resmi Kelompok 3 (pas 1 halaman)
│   ├── Abstrak-Indo.tex                              ← Abstrak Bahasa Indonesia & kata kunci (pas 1 halaman)
│   ├── Pendahuluan.tex                               ← Bab I: Latar Belakang, Rumusan Masalah, Tujuan, Ruang Lingkup
│   ├── Kajian-Pustaka.tex                            ← Bab II: Landasan Teori UBCF, IBCF, MF, Metrik & 5-Fold CV
│   ├── Metodologi.tex                                ← Bab III: Diagram Alir TikZ, Spesifikasi Data & Skenario Pengujian
│   ├── Hasil-Pembahasan.tex                          ← Bab IV: Analisis Hasil EDA, Baseline, Model CF, Uji Statistik, Top-10
│   ├── Kesimpulan.tex                                ← Bab V: Kesimpulan menjawab 4 rumusan masalah & saran
│   ├── References.bib                                ← Daftar pustaka terstandar BibTeX
│   ├── Lampiran.tex                                  ← Lampiran: Log hyperparameter tuning MF & rekapitulasi kendala teknis
│   └── Tel-U-Logo.png                                ← Logo resmi Universitas Telkom untuk halaman judul
├── src/                                              ← Sumber logika modular (Single Source of Truth)
│   ├── data.py                                       ← Pemuat data, verifikasi SHA-256, matriks rating
│   ├── metrics.py                                    ← MAE, MSE, RMSE, NMSE, DCG, NDCG@k
│   ├── baseline.py                                   ← Model acuan (Random, GlobalMean, UserMean, ItemMean)
│   ├── ubcf.py                                       ← User-Based CF (Pearson, significance weighting, k-NN)
│   ├── ibcf.py                                       ← Item-Based CF (Cosine, Adjusted Cosine, k-NN)
│   ├── mf.py                                         ← Matrix Factorization (SGD, regularisasi L2)
│   ├── cache.py                                      ← Caching deterministik hasil komputasi berat
│   ├── experiment.py                                 ← Runner 5-fold CV paralel, sensitivitas, rekomendasi
│   ├── report.py                                     ← Formatter tabel & gambar bernomor otomatis
│   └── plot_style.py                                 ← Konfigurasi gaya visualisasi grafik laporan
├── scripts/
│   └── mf_tuning.py                                  ← Penalaan hyperparameter MF pada data validasi terpisah
├── tests/                                            ← Unit test rumus & fungsi terhadap contoh hitungan slide
│   ├── test_metrics.py                               ← Pengujian MAE, MSE, RMSE, NMSE, NDCG (5 test)
│   ├── test_ubcf.py                                  ← Pengujian contoh Alice Slide M02 (3 test)
│   ├── test_ibcf.py                                  ← Pengujian contoh film & Alice Slide M03 (4 test)
│   ├── test_mf.py                                    ← Pengujian contoh matriks 5x4 Slide M04 (3 test)
│   └── test_baseline.py                              ← Pengujian logika keempat baseline (4 test)
├── figures/                                          ← Seluruh grafik laporan (fig_*.png, 11 figur resolusi tinggi)
├── results/                                          ← Berkas rekapitulasi evaluasi (CSV)
│   ├── perbandingan_rata2.csv                        ← Rata-rata 5-fold seluruh metode
│   ├── perbandingan_sd.csv                           ← Simpangan baku 5-fold seluruh metode
│   ├── uji_berpasangan.csv                           ← Uji-t berpasangan (paired t-test)
│   ├── perbaikan_vs_baseline.csv                     ← Persentase peningkatan performa
│   └── waktu_komputasi.csv                           ← Rekap waktu pelatihan & inferensi
├── PROGRESS.md                                       ← Log perkembangan pengerjaan dan keputusan teknis
├── MASALAH_DAN_PENYELESAIAN.md                       ← Rekap kendala teknis dan solusi penerapannya
└── requirements.txt                                  ← Dependensi pustaka Python
```

---

## Petunjuk Menjalankan

### 1. Penyiapan Lingkungan Virtual

```bash
# Buat virtual environment
python -m venv .venv

# Aktivasi di Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Pasang dependensi
pip install -r requirements.txt
```

### 2. Menjalankan Unit Test (Verifikasi Rumus Sesuai Slide)

```bash
.venv\Scripts\python tests\test_metrics.py
.venv\Scripts\python tests\test_ubcf.py
.venv\Scripts\python tests\test_ibcf.py
.venv\Scripts\python tests\test_mf.py
.venv\Scripts\python tests\test_baseline.py
```
*(Seluruh 19 unit test terkonfirmasi **LULUS**).*

### 3. Menjalankan Notebook Utama

Buka `notebooks/Evaluasi_RecSys_CF_MF.ipynb` pada Jupyter Notebook / VS Code, lalu pilih **Run All**.
- Berkat sistem caching (`results/cache/`) dan paralelisasi Matrix Factorization (`joblib`), eksekusi penuh selesai dalam **< 1 menit** (44,8 detik) di CPU.
- Seluruh grafik disimpan otomatis ke `figures/` dan metrik disimpan ke `results/`.
- Notebook berfokus murni pada aspek teknis praktis (pipeline data, visualisasi EDA, sanity check metrik, evaluasi 5-fold CV, tabel, grafik, insight teknis).

### 4. Mengompilasi Dokumen Laporan LaTeX Formal

Laporan akademik formal tersimpan secara terstruktur dan terisolasi di dalam folder `laporan/`. Untuk mengompilasi ulang menjadi `main.pdf`:

```bash
cd "laporan"
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```
Dokumen `laporan/main.pdf` (42 halaman) mencakup Cover resmi Kelompok 3 (tanpa lembar pengesahan), Abstrak, Daftar Isi/Gambar/Tabel, Bab I Pendahuluan, Bab II Kajian Pustaka, Bab III Metodologi TikZ, Bab IV Hasil & Pembahasan Komparatif, Bab V Kesimpulan, Daftar Pustaka BibTeX, serta Lampiran penalaan parameter & kendala teknis.

---

## Hasil Evaluasi Komparatif (Rata-rata 5-Fold Cross Validation)

| Metode | MAE | MSE | RMSE | NMSE | NDCG@10 |
|:---|:---:|:---:|:---:|:---:|:---:|
| *Baseline: Tebak acak* | 1,3902 | 2,8901 | 1,6999 | 0,7225 | 0,6839 |
| *Baseline: Rata-rata global* | 0,9447 | 1,2672 | 1,1256 | 0,3168 | 0,7366 |
| *Baseline: Rata-rata pengguna* | 0,8362 | 1,0895 | 1,0437 | 0,2724 | 0,7366 |
| *Baseline: Rata-rata film* | 0,8174 | 1,0498 | 1,0246 | 0,2624 | 0,8329 |
| **User-Based CF** (Pearson, $k=30, \gamma=50$) | **0,7342** | 0,8901 | 0,9434 | 0,2225 | 0,8474 |
| **Item-Based CF** (Adjusted Cosine, $k=30, \gamma=50$) | 0,7383 | 0,8949 | 0,9459 | 0,2237 | 0,8375 |
| **Matrix Factorization** (SGD, $K=100, \beta=0{,}2$) | 0,7357 | **0,8554** | **0,9249** | **0,2138** | **0,8578** |

### Temuan Utama:
1. **Reduksi Galat Nyata**: Ketiga algoritma memangkas galat RMSE sebesar 7,5%–9,7% dibandingkan baseline terkuat (`Rata-rata film`).
2. **Kemenangan Matrix Factorization**: MF mencatatkan RMSE terendah (0,9249) dan peringkat NDCG@10 tertinggi (0,8578), konsisten menang di seluruh 5 fold ($p < 0{,}01$, uji-$t$ berpasangan).
3. **Ketahanan terhadap Sparsity**: Item-Based CF paling rentan mengalami degradasi akurasi pada film-film bervolume rating rendah (*long-tail*), sementara representasi ruang laten MF memberikan ketahanan generalisasi tertinggi.

---

## Referensi

- Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets: History and Context. *ACM TiiS*, 5(4), Article 19. https://doi.org/10.1145/2827872
- Jannach, D., Zanker, M., Felfernig, A., & Friedrich, G. (2010). *Recommender Systems: An Introduction*. Cambridge University Press.
- Järvelin, K., & Kekäläinen, J. (2002). Cumulated gain-based evaluation of IR techniques. *ACM TOIS*, 20(4), 422–446.
- Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix Factorization Techniques for Recommender Systems. *Computer*, 42(8), 30–37.
- Sarwar, B., Karypis, G., Konstan, J., & Riedl, J. (2001). Item-based collaborative filtering recommendation algorithms. *Proc. 10th WWW*, 285–295.
- Slide Perkuliahan Sistem Pemberi Rekomendasi (Sub-CLO 12-1-1 Evaluasi), Prof. Dr. Z.K. Abdurahman Baizal, S.Si., M.Kom. & Agung Toto Wibowo, S.T., M.T.
