# Log Progres & Keputusan — Tugas Kelompok RecSys (Kelompok 3)

File ini mencatat setiap langkah yang sudah dikerjakan beserta alasan di balik setiap pilihan.
Gunanya dua: (1) bahan untuk bab "Data dan Metode" di laporan, (2) supaya semua anggota kelompok
bisa paham kenapa sesuatu dikerjakan dengan cara tertentu.

Data anggota kelompok: lihat [README.md](README.md).

---

## Ringkasan Status

| Tahap | Status |
|---|---|
| Membaca slide materi & menentukan rumus | Selesai (9 Nov 2026) |
| Memilih & mengunduh dataset | Selesai (9 Nov 2026) |
| Eksplorasi data (EDA) | Selesai (9 Nov 2026), `notebooks/01_eda.ipynb` |
| Fungsi metrik + pengujian dengan contoh slide | Selesai (9 Nov 2026), `src/metrics.py` |
| User-based CF | Selesai (9 Nov 2026), `notebooks/02_user_based_cf.ipynb` |
| Item-based CF | Selesai (9 Nov 2026), `notebooks/03_item_based_cf.ipynb` |
| Matrix Factorization | Selesai (9 Nov 2026), `notebooks/04_matrix_factorization.ipynb` |
| Perbandingan akhir ketiga metode | Selesai (9 Nov 2026), `notebooks/05_perbandingan.ipynb` |
| Repo GitHub + README | Push pertama 9 Nov 2026 (UBCF, IBCF, kode MF); repo: github.com/alvitobuana/Tugas-Kelompok-Sub-CLO-12-1-1-Evaluasi |
| Share repo ke dosen | Repo dibiarkan public (keputusan 9 Nov 2026), jadi dosen bisa membuka link tanpa undangan |
| Laporan PDF | Belum |

---

## 2026-11-09 — Membaca slide materi dosen

**Yang dilakukan:** membaca semua slide di folder materi (Pengantar RecSys, M02 User-Based CF,
M03 Item-Based CF, M04 Matrix Factorization, Evaluasi Part 1, Evaluasi Part 2, Evaluasi tambahan).

**Kesimpulan:** semua rumus yang dibutuhkan tugas tersedia di slide. Rumus yang dipakai:

| Bagian | Rumus / pengaturan yang dipakai | Sumber slide |
|---|---|---|
| User-based CF | Similarity Pearson correlation, prediksi = rata-rata user aktif + rata-rata tertimbang selisih rating tetangga terhadap rata-ratanya. Jumlah tetangga k di kisaran 20–50 | M02 |
| Item-based CF | Cosine similarity dan adjusted cosine similarity (rating dikurangi rata-rata user), prediksi = rata-rata tertimbang rating user pada item-item yang mirip, dengan threshold/jumlah tetangga | M03 |
| Matrix Factorization | Faktorisasi R ≈ P × Qᵀ dengan k latent feature, dilatih memakai Stochastic Gradient Descent | M04 |
| MAE, MSE, RMSE | Rumus standar seperti di slide | Evaluasi Part 1 |
| NMSE | NMSE = MSE / (r_max − r_min) | Evaluasi Part 1 |
| NDCG | NDCG@l = DCG@l / IDCG@l, dengan DCG@l = Σ rel_i / log₂(i+1) | Evaluasi Part 2 |

**Catatan penting tentang slide:**

1. **Rumus NDCG di slide tidak konsisten.** DCG ditulis dengan `rel_i`, sedangkan IDCG ditulis
   dengan `2^rel_i − 1`. Jika keduanya dicampur, NDCG bisa bernilai lebih dari 1, padahal
   seharusnya berada di rentang 0–1. **Keputusan:** DCG dan IDCG sama-sama memakai `rel_i`
   (bentuk linear), karena bentuk inilah yang dipakai di contoh hitungan pada slide yang sama.
   Hal ini akan disebutkan di laporan.
2. **Typo di slide NMSE:** baris kedua tertulis "rmax is the minimum rating", maksudnya r_min.
   Tidak berpengaruh ke perhitungan.
3. **Batas relevansi untuk NDCG:** item dianggap relevan jika rating ≥ 4. Aturan ini diambil dari
   soal Kuis 2 di slide Evaluasi Part 2, supaya sama dengan cara dosen.

**Alasan mengikuti slide secara ketat:** soal tugas meminta pengujian "menggunakan metriks yang ada
dalam slide". Beberapa metrik (terutama NMSE dan NDCG) punya beberapa versi berbeda di literatur,
sehingga memakai versi dosen adalah pilihan paling aman.

---

## 2026-11-09 — Memilih dataset

**Keputusan:** MovieLens 100K (GroupLens Research, University of Minnesota).

**Kandidat yang dibandingkan:**

| Dataset | Ukuran | Alasan tidak/dipilih |
|---|---|---|
| **MovieLens 100K** | 100.000 rating, 943 user, 1.682 film | **Dipilih**, alasan di bawah |
| MovieLens Latest Small | ±100.000 rating, 610 user, ±9.700 film | Skala setengah bintang (0.5–5) dan matriks lebih jarang terisi |
| Book-Crossing | ±1 juta rating | Banyak rating 0 yang sebenarnya berarti "tidak dirating", perlu pembersihan berat, sangat sparse |
| Anime Recommendations (Kaggle) | ±7,8 juta rating | Terlalu besar untuk CF yang ditulis manual; nilai −1 berarti "ditonton tapi tidak dirating" |
| Goodbooks-10k | ±6 juta rating | Terlalu besar untuk user-based CF (matriks similarity ±53 ribu × 53 ribu user) |

**Alasan memilih MovieLens 100K:**

1. **Memenuhi syarat "ada user history".** Setiap baris berisi user ID, item ID, rating, dan
   timestamp. Setiap user memiliki minimal 20 rating.
2. **Ukuran pas untuk implementasi manual.** Matriks rating 943 × 1.682, matriks similarity user
   943 × 943, dan matriks similarity item 1.682 × 1.682 masih ringan untuk dihitung di laptop.
3. **Sejalan dengan slide.** Slide M02 mengutip hasil analisis dataset MovieLens untuk pemilihan
   jumlah tetangga (k = 20–50). Dataset ini juga sudah menyediakan pembagian 5-fold
   (`u1`–`u5`), sesuai dengan materi 5-fold cross validation di slide Evaluasi Part 1.
4. **Skala rating 1–5 bilangan bulat**, sehingga NMSE cukup dibagi (5 − 1) = 4 dan aturan
   "relevan jika rating ≥ 4" langsung bisa diterapkan.
5. **Dataset standar** di bidang sistem rekomendasi, sehingga hasil kami mudah dibandingkan
   dengan buku dan paper.

**Kekurangan yang disadari:** datanya lama (rating tahun 1997–1998), tetapi hal ini tidak
memengaruhi tujuan tugas, yaitu membandingkan tiga metode.

---

## 2026-11-09 — Mengunduh dan memeriksa dataset

**Sumber:** https://files.grouplens.org/datasets/movielens/ml-100k.zip (resmi dari GroupLens)
**Lokasi:** `data/ml-100k.zip`, diekstrak ke `data/ml-100k/`
**SHA-256 zip:** `50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229`

**File yang akan dipakai:**

| File | Isi |
|---|---|
| `u.data` | 100.000 rating: user, item, rating, timestamp (dipisah tab) |
| `u.item` | Judul film dan genre (untuk menampilkan hasil rekomendasi) |
| `u.user` | Data demografis user (hanya untuk deskripsi, tidak dipakai di model) |
| `u1.base` … `u5.base` / `u1.test` … `u5.test` | Pembagian 5-fold bawaan, masing-masing 80.000 train / 20.000 test |

**Hasil pemeriksaan data:**

| Statistik | Nilai |
|---|---|
| Jumlah rating | 100.000 |
| Jumlah user | 943 |
| Jumlah film | 1.682 |
| Kepadatan matriks (density) | 6,30% (artinya 93,7% sel kosong → sparse) |
| Rating per user | min 20, median 65, maks 737 |
| Rating per film | min 1, median 27, maks 583 |
| Rata-rata rating | 3,53 |
| Distribusi rating | 1: 6.110 · 2: 11.370 · 3: 27.145 · 4: 34.174 · 5: 21.201 |
| Rating relevan (≥ 4) | 55,38% |
| Rentang waktu | 20 Sep 1997 – 22 Apr 1998 |
| Data kosong / duplikat user-item | 0 / 0 |

Cara menghitung density:

    density = jumlah rating / (jumlah user × jumlah film)
            = 100.000 / (943 × 1.682)
            = 100.000 / 1.586.126
            ≈ 0,0630 → 6,30%

**Interpretasi singkat:** data sudah bersih (tanpa nilai kosong dan duplikat), jadi tidak perlu
pembersihan. Tantangan utamanya adalah sparsity: banyak film hanya dirating sedikit user, sehingga
similarity antar-item bisa kurang andal untuk film yang jarang ditonton. Hal ini nanti relevan saat
membahas perbedaan hasil user-based dan item-based CF.

---

## 2026-11-09 — Menyiapkan lingkungan kerja dan struktur project

**Yang dilakukan:** membuat virtual environment `.venv` khusus project (agar Python utama di laptop
tidak ikut berubah) dan memasang library dari `requirements.txt`, termasuk Jupyter untuk notebook.

**Struktur project:**

| Folder/file | Isi |
|---|---|
| `src/data.py` | Membaca dataset dan mengubahnya menjadi matriks rating user × film |
| `src/metrics.py` | MAE, MSE, RMSE, NMSE, NDCG@l sesuai rumus slide |
| `src/ubcf.py` | User-based CF |
| `src/ibcf.py` | Item-based CF |
| `src/baseline.py` | Baseline pembanding (acak, rata-rata global/user/film) |
| `src/mf.py` | Matrix factorization (SGD) |
| `scripts/mf_tuning.py` | Pemilihan hyperparameter MF dengan data validasi |
| `src/plot_style.py` | Gaya grafik laporan |
| `tests/` | Pengujian kode memakai contoh hitungan di slide |
| `notebooks/` | Notebook langkah demi langkah (bahan screenshot laporan) |
| `figures/` | Grafik untuk laporan |
| `results/` | Tabel hasil evaluasi (CSV) |

**Alasan memisah kode (`src/`) dan notebook:** rumus cukup ditulis sekali di `src/`, lalu dipakai ulang
oleh ketiga notebook dan oleh pengujian. Notebook fokus pada penjelasan langkah dan hasil.

**Alasan memakai Jupyter Notebook:**
1. Soal tugas meminta penjelasan langkah dan screenshot source code yang dijelaskan. Notebook menampilkan
   penjelasan, kode, dan hasil dalam satu tempat, sehingga satu screenshot sudah memuat kode dan hasilnya.
2. Sejalan dengan slide M04 yang meminta kode MF dijalankan di Google Colab (Jupyter Notebook versi online).
3. Output dan grafik tersimpan di dalam notebook, sehingga bisa dibaca langsung di GitHub tanpa dijalankan.

Kelemahan notebook diantisipasi dengan: rumus inti di `src/` (bisa dipakai ulang), pengujian di `tests/`,
semua notebook dijalankan ulang dari sel pertama sampai terakhir sebelum disimpan (hasil konsisten), dan
proses berat (tuning MF) dipisah ke `scripts/`.

---

## 2026-11-09 — Fungsi metrik evaluasi

**Yang dilakukan:** menulis `src/metrics.py` dan mengujinya (`tests/test_metrics.py`) dengan contoh
hitungan di slide. Semua 5 test lulus:

| Contoh di slide | Harapan | Hasil kode |
|---|---|---|
| MAE contoh UserCF (Evaluasi tambahan) | 1,75 | 1,75 |
| CG contoh NDCG (Evaluasi Part 2) | 11 | 11 |
| DCG₆ | 6,861 | 6,861 |
| IDCG₆ | 7,141 | 7,141 |
| nDCG₆ | 0,961 | 0,961 |

**Keputusan cara menghitung NDCG@10 pada data uji:**

1. Untuk setiap user, film-film miliknya di data uji diurutkan berdasarkan rating prediksi (terbesar
   dulu). Jika prediksi sama, diurutkan berdasarkan id film supaya hasil selalu sama.
2. Relevansi biner: 1 jika rating asli ≥ 4, selain itu 0 (aturan Kuis 2).
3. NDCG dihitung pada 10 posisi teratas, lalu dirata-rata untuk semua user.
4. User yang tidak punya film relevan di data uji dilewati, karena IDCG = 0 sehingga NDCG tidak
   terdefinisi (pembagian dengan nol).

**Catatan bagus untuk laporan:** karena relevansinya biner (0 atau 1), ketidakkonsistenan rumus di slide
(`rel_i` vs `2^rel_i − 1`) **tidak berpengaruh**, sebab 2¹ − 1 = 1 dan 2⁰ − 1 = 0. Kedua rumus memberi hasil
yang sama persis. Selain itu, contoh hitungan di slide sendiri memakai `rel_i` linear untuk DCG maupun IDCG.

**Alasan memilih "mengurutkan film di data uji" untuk NDCG:** rating asli hanya diketahui untuk film di
data uji. Film lain yang belum pernah dirating user tidak bisa dinilai relevan atau tidak. Cara ini juga
sama dengan soal Kuis 2, yang menghitung NDCG dari film-film yang rating aslinya diketahui.

---

## 2026-11-09 — Eksplorasi data (notebook 01)

**Yang dilakukan:** `notebooks/01_eda.ipynb` berisi statistik dasar, perhitungan density, dan 3 grafik
(`figures/eda_distribusi_rating.png`, `eda_rating_per_user.png`, `eda_rating_per_film.png`).

**Temuan tambahan:**
- 333 dari 1.682 film (19,8%) dirating kurang dari 5 kali → *long tail*, sulit diprediksi.
- Di setiap fold ada 22–34 film di data uji yang tidak muncul sama sekali di data latih (masalah
  *cold start* untuk item). Untuk film seperti ini, item-based CF dan MF nanti perlu nilai cadangan.
- Jumlah user di data uji berbeda per fold (459 di fold 1, 927 di fold 5), karena MovieLens membagi
  fold berdasarkan urutan baris di `u.data` (skrip `mku.sh`: test fold ke-i = baris ke-(20.000·(i−1)+1)
  sampai 20.000·i), bukan per user. Ini wajar dan tetap valid karena kelima test set tidak saling
  beririsan, sesuai README MovieLens ("disjoint test sets ... for 5 fold cross validation").

---

## 2026-11-09 — User-based CF (notebook 02)

**Implementasi:** `src/ubcf.py`, diuji dengan contoh Alice di slide M02 (`tests/test_ubcf.py`):

| Contoh Alice | Slide | Kode |
|---|---|---|
| sim(Alice, User1) | 0,85 | 0,85 |
| sim(Alice, User2) | 0,70 | 0,71 (nilai pastinya 0,7071; slide membulatkan ke bawah) |
| sim(Alice, User3) | 0,00 | 0,00 |
| sim(Alice, User4) | −0,79 | −0,79 |
| Prediksi Alice untuk Item5 (k = 2) | 4,87 | 4,87 |

**Keputusan dan alasannya:**

| Keputusan | Alasan |
|---|---|
| Pearson dihitung hanya pada film co-rated, dengan rata-rata di film co-rated | Sama dengan contoh Alice di slide (0,85 hanya keluar jika memakai cara ini) |
| Prediksi memakai rata-rata user dari semua rating miliknya | Sama dengan contoh slide (4 + ... memakai rata-rata 2,4 dan 3,8) |
| Hanya tetangga dengan similarity positif | Slide M02: "we included only those that had a positive correlation" |
| k = 30 | Slide M02: 20–50 tetangga wajar untuk MovieLens; 30 diambil di tengah |
| Significance weighting, γ = 50 | Slide M02 bagian "Improving the metrics". Tanpa ini, tetangga teratas user 1 adalah user yang hanya punya 2–3 film sama dengan korelasi 1,0 |
| Pasangan dengan < 2 film co-rated diberi similarity 0 | Pearson tidak terdefinisi dengan 1 titik data |
| Jika tidak ada tetangga, prediksi = rata-rata user | Sama dengan cara cadangan (default back-off) di slide Evaluasi tambahan |
| Prediksi dipotong ke rentang 1–5 | Rating di luar 1–5 tidak mungkin terjadi |
| Untuk rekomendasi top-10, film harus dirating minimal 5 tetangga | Tanpa syarat ini, top-10 user 1 dikuasai film obscure berprediksi 5,0 yang hanya didukung 1–3 tetangga (misalnya *Santa with Muscles*). Syarat ini **tidak** dipakai saat evaluasi metrik |

**Tentang pemilihan k:** k dan γ ditetapkan dari slide **sebelum** melihat hasil. Tabel sensitivitas
di notebook hanya untuk analisis, bukan untuk memilih k terbaik dari data uji (memilih parameter
berdasarkan data uji membuat hasil terlihat lebih bagus dari kenyataan).

**Hasil sensitivitas (fold 1):** significance weighting menurunkan RMSE di semua nilai k
(misalnya k = 30: 0,9653 → 0,9536). Hasil cukup stabil untuk k = 20–50, sesuai saran slide.

**Hasil 5-fold (k = 30, γ = 50):**

| Metrik | Rata-rata | Simpangan baku |
|---|---|---|
| MAE | 0,7342 | 0,0047 |
| MSE | 0,8901 | 0,0114 |
| RMSE | 0,9434 | 0,0060 |
| NMSE | 0,2225 | 0,0029 |
| NDCG@10 | 0,8474 | 0,0088 |

Interpretasi singkat: rata-rata prediksi meleset sekitar 0,73 bintang dari rating asli. Simpangan baku
antar-fold kecil, artinya hasilnya stabil dan tidak bergantung pada satu pembagian data saja.

**Contoh rekomendasi untuk user 1** (dilatih dengan seluruh data): Pather Panchali, A Close Shave,
Casablanca, Faust, Schindler's List, L.A. Confidential, Secrets & Lies, Rear Window, Dr. Strangelove,
Lawrence of Arabia.

---

## 2026-11-09 — Temuan data: judul film ganda

Saat menampilkan rekomendasi item-based CF, film *Ulee's Gold (1997)* muncul dua kali. Setelah diperiksa,
MovieLens 100K memang punya **18 judul yang terdaftar dua kali dengan id berbeda** (misalnya *Chasing Amy*
= id 246 dan 268), mencakup 1.269 rating (1,27%).

**Keputusan:** id ganda tetap dibiarkan terpisah saat evaluasi, karena pembagian fold resmi MovieLens dibuat
berdasarkan id dan jumlahnya kecil. Saat menampilkan rekomendasi, fungsi `unique_titles` di `src/data.py`
menyaring judul yang sama agar tidak muncul dua kali, dan judul yang sudah pernah dirating user melalui id
kembarannya tidak direkomendasikan lagi. Temuan ini ditambahkan ke notebook 01 (bagian 6).

---

## 2026-11-09 — Item-based CF (notebook 03)

**Implementasi:** `src/ibcf.py`, diuji dengan contoh di slide M03 (`tests/test_ibcf.py`, 4 test lulus):

| Contoh di slide | Slide | Kode |
|---|---|---|
| Cosine Pulp Fiction vs Forrest Gump | 0,792 | 0,7928 |
| Cosine Pulp Fiction vs The GodFather | 0,902 | 0,902 |
| Prediksi Amy untuk Forrest Gump: (4·0,792 + 5·0,8) / (0,792 + 0,8) | 4,5 | 4,5 |
| Cosine biasa sim(I5, I1), contoh Alice | 0,99 | 0,99 |
| Adjusted cosine sim(I5, I1), contoh Alice | 0,80 | 0,80 |

**Catatan slide:** tabel similarity film di slide menulis Forrest Gump vs The GodFather = 0,8, padahal jika
dihitung dari user yang merating keduanya (Robert 3 dan 1, David 2 dan 2) hasilnya 7 / (√13 · √5) = 0,868.
Kemungkinan salah ketik atau pembulatan di slide. Kode menghasilkan 0,868. Angka ini dijelaskan di notebook.

**Keputusan dan alasannya:**

| Keputusan | Alasan |
|---|---|
| Cosine dan adjusted cosine dihitung hanya pada user yang merating kedua film | Sesuai contoh Pulp Fiction (hanya Calvin, Robert, Bradley) dan contoh Alice di slide |
| Rata-rata user untuk adjusted cosine = rata-rata seluruh rating user | Sesuai contoh Alice (rata-rata User1 = 2,4 dari 5 rating) |
| Prediksi = Σ sim · rating / Σ sim memakai rating mentah | Rumus prediksi slide M03 |
| Hanya film dengan similarity positif yang dipakai | Bobot negatif membuat rata-rata tertimbang tidak bermakna (bisa keluar dari rentang 1–5). Ini juga bentuk threshold similarity yang disebut slide |
| Konfigurasi akhir: adjusted cosine, k = 30, γ = 50 | Adjusted cosine disarankan slide; k = 30 mengikuti ukuran tetangga yang disarankan Sarwar et al. (2001); γ = 50 disamakan dengan UBCF agar perbandingan adil. Ditetapkan sebelum melihat hasil |
| Jika tidak ada film mirip yang dirating user → rata-rata user | Terjadi 29–38 kali per fold (sekitar 0,18%), terutama untuk film yang tidak ada di data latih |
| Rekomendasi: minimal 5 film pendukung, judul ganda disaring | Sama seperti UBCF |

**Hasil perbandingan varian similarity (fold 1, k = 30):**

| Varian | MAE | RMSE | NDCG@10 |
|---|---|---|---|
| Cosine biasa | 0,8283 | 1,0508 | 0,7510 |
| Cosine biasa + significance weighting | 0,8556 | 1,1003 | 0,7674 |
| Adjusted cosine | 0,7851 | 0,9966 | 0,8210 |
| **Adjusted cosine + significance weighting** | **0,7523** | **0,9644** | **0,8308** |

**Analisis untuk laporan:**
1. **Adjusted cosine jauh lebih baik daripada cosine biasa** (RMSE turun dari 1,0508 ke 0,9966). Karena semua
   rating bernilai positif, cosine biasa menilai hampir semua film "mirip". Contohnya, 5 film teratas
   yang mirip Star Wars semuanya bernilai 0,979–0,989, sehingga similarity hampir tidak membedakan film
   yang benar-benar mirip dari yang tidak. Adjusted cosine memberi rentang yang jauh lebih lebar
   (0,63–0,82) dan bisa bernilai negatif.
2. **Significance weighting membantu adjusted cosine, tetapi memperburuk cosine biasa.** Dugaan kami:
   karena nilai cosine biasa hampir sama semua, setelah dikalikan bobot jumlah user co-rated, urutan
   tetangga praktis hanya ditentukan oleh popularitas film, bukan kemiripan selera. Ini dugaan dan belum
   kami buktikan secara terpisah.

**Hasil 5-fold (adjusted cosine, k = 30, γ = 50):**

| Metrik | Rata-rata | Simpangan baku |
|---|---|---|
| MAE | 0,7383 | 0,0086 |
| MSE | 0,8949 | 0,0207 |
| RMSE | 0,9459 | 0,0109 |
| NMSE | 0,2237 | 0,0052 |
| NDCG@10 | 0,8375 | 0,0064 |

Dibanding user-based CF (MAE 0,7342, RMSE 0,9434, NDCG@10 0,8474), item-based CF sedikit lebih buruk di
semua metrik, tetapi selisihnya kecil (MAE berbeda 0,004). Selisih ini masih lebih kecil daripada simpangan
baku antar-fold, sehingga belum bisa dikatakan bahwa satu metode jelas lebih unggul. Ini sejalan dengan
slide M02 yang menyebut perbedaan antar metode CF "often very small (1/100)".

**Contoh rekomendasi untuk user 1:** Paris Is Burning, Ulee's Gold, Like Water for Chocolate, When We Were
Kings, The Double Life of Véronique, Casablanca, Duck Soup, Brassed Off, My Life as a Dog, Kundun.
Film yang paling mirip dengan Star Wars menurut model: The Empire Strikes Back, Return of the Jedi,
Raiders of the Lost Ark. Hasil ini masuk akal dan menunjukkan kelebihan item-based CF yang mudah
dijelaskan ("karena Anda menyukai film X").

---

## 2026-11-09 — Matrix factorization (notebook 04)

**Implementasi:** `src/mf.py`, diuji dengan contoh di slide M04 (`tests/test_mf.py`, 3 test lulus).

**Rumus (sama dengan slide):**

    e_ij  = r_ij − p_i · q_j
    p_ik ← p_ik + α (2 e_ij q_kj − β p_ik)
    q_kj ← q_kj + α (2 e_ij p_ik − β q_kj)

**Uji dengan contoh slide:** matriks 5 user × 4 film, K = 2, α = 0,0002, β = 0,02, 5.000 iterasi, inisialisasi
acak 0–1 (persis kode slide). RMSE pada rating yang diketahui turun dari 2,440 ke 0,040, dan pola hasilnya
sama dengan slide (U1–U2 suka D1, U3–U5 suka D4). Angka prediksi sel kosong sedikit berbeda dari slide
karena angka acak awalnya berbeda.

**Perbedaan kecil dengan kode slide dan alasannya:**

| Kode slide | Implementasi kami | Alasan |
|---|---|---|
| Rating dilalui berurutan | Urutan rating diacak tiap epoch | Inilah sifat "stochastic" pada SGD; urutan tetap bisa membuat model bias ke user/film yang selalu diproses terakhir |
| p diupdate dulu, lalu q memakai p yang sudah baru | p dan q diupdate serentak dari nilai lama | Sesuai turunan rumus di slide (kedua gradien dihitung di titik yang sama) |
| Acak 0–1 | Acak 0–s, dengan s = 2·√(rata-rata rating / K) | Dengan K besar, acak 0–1 membuat prediksi awal jauh di atas 5 (rata-rata K/4). Penskalaan ini membuat prediksi awal ≈ rata-rata rating. Contoh slide tetap memakai acak 0–1 |
| Berhenti jika error < 0,001 | Berhenti di jumlah epoch yang dipilih lewat data validasi | Pada data asli error latih tidak akan pernah < 0,001 tanpa overfitting parah |
| Film tanpa rating latih tetap diprediksi p · q | Diganti rata-rata user | Vektor q film tersebut tidak pernah dilatih (masih acak) |

**Pemilihan hyperparameter (`scripts/mf_tuning.py`):** data latih fold 1 dibagi 90% latih (72.106 rating) dan
10% validasi (7.894 rating). Data uji tidak dipakai sama sekali. Pencarian dilakukan bertahap:

1. Learning rate (K = 20, β = 0,02): α = 0,0002 (nilai slide) masih terus membaik sampai epoch 60 (RMSE
   validasi 0,9647), jadi terlalu lambat. α = 0,005 dan 0,01 cepat turun, tetapi langsung overfitting
   setelah epoch 3–6 karena β = 0,02 terlalu lemah.
2. K ∈ {5, 10, 20, 50} × β ∈ {0,02; 0,1}: β = 0,1 lebih baik di semua K. Terbaik K = 50, β = 0,1, tetapi
   **berada di ujung grid**.
3. Perluasan β = 0,2 dan K = 100: terbaik K = 100, β = 0,2, masih di ujung grid.
4. Perluasan kedua β ∈ {0,3; 0,5}, K = 200, maksimal 100 epoch: β = 0,3 dan K = 200 sama-sama lebih buruk,
   sehingga optimum sudah berada di tengah grid dan pencarian dihentikan.

RMSE validasi terbaik per kombinasi (α = 0,01):

| | β = 0,02 | β = 0,1 | β = 0,2 | β = 0,3 | β = 0,5 |
|---|---|---|---|---|---|
| K = 5 | 0,9621 | 0,9439 | | | |
| K = 10 | 0,9796 | 0,9580 | | | |
| K = 20 | 0,9732 | 0,9543 | 0,9276 | | |
| K = 50 | 0,9663 | 0,9413 | 0,9185 | 0,9249 | |
| K = 100 | | 0,9274 | **0,9140** | 0,9257 | 0,9686 |
| K = 200 | | | 0,9157 | 0,9262 | |

**Konfigurasi akhir:** K = 100, α = 0,01, β = 0,2, 35 epoch (RMSE validasi 0,9140).

**Alasan memperluas grid:** jika nilai terbaik ada di ujung rentang yang dicoba, ada kemungkinan nilai yang
lebih baik berada di luar rentang tersebut. Grid diperluas sampai nilai terbaik diapit oleh nilai yang lebih
buruk di kedua sisi.

**Temuan untuk analisis:**
- Dengan β = 0,02 (nilai slide), RMSE validasi **naik terus** setelah epoch 3–6 → overfitting.
  Dengan β = 0,2, kurva validasi mendatar setelah sekitar epoch 20 dan tidak naik. Grafik:
  `figures/mf_learning_rate.png` dan `figures/mf_kurva_belajar.png`.
- K besar (100) hanya bermanfaat jika regularisasinya cukup kuat. Dengan β = 0,02, K = 50 justru lebih
  buruk daripada K = 5.

**Hasil 5-fold (K = 100, α = 0,01, β = 0,2, 35 epoch):**

| Metrik | Rata-rata | Simpangan baku |
|---|---|---|
| MAE | 0,7357 | 0,0052 |
| MSE | 0,8554 | 0,0097 |
| RMSE | 0,9249 | 0,0052 |
| NMSE | 0,2138 | 0,0024 |
| NDCG@10 | 0,8578 | 0,0057 |

**Contoh rekomendasi untuk user 1** (film minimal dirating 5 user, judul ganda disaring): A Close Shave,
Pather Panchali, The Third Man, Faust, Lawrence of Arabia, Secrets & Lies, Dr. Strangelove, Ran,
Down by Law, Rosencrantz and Guildenstern Are Dead.

---

## 2026-11-09 — Perbandingan sementara ketiga metode (rata-rata 5-fold)

| Metode | MAE | MSE | RMSE | NMSE | NDCG@10 |
|---|---|---|---|---|---|
| User-based CF | **0,7342** | 0,8901 | 0,9434 | 0,2225 | 0,8474 |
| Item-based CF | 0,7383 | 0,8949 | 0,9459 | 0,2237 | 0,8375 |
| Matrix factorization | 0,7357 | **0,8554** | **0,9249** | **0,2138** | **0,8578** |

**Pengamatan awal:** MF unggul jelas di MSE, RMSE, NMSE, dan NDCG@10, tetapi MAE-nya sedikit di atas UBCF.
Karena MSE/RMSE mengkuadratkan error, metrik ini sangat menghukum kesalahan besar. Artinya MF lebih jarang
membuat kesalahan besar (misalnya menebak 2 padahal aslinya 5), meskipun rata-rata kesalahan absolutnya
mirip dengan UBCF. Analisis ini akan diperdalam di notebook 05 (sebaran error, uji per fold, dan baseline).

---

## Lingkungan kerja

| Komponen | Versi |
|---|---|
| Python | 3.13.1 (virtual environment `.venv`) |
| NumPy | 2.2.3 |
| Pandas | 3.0.5 |
| scikit-learn | 1.9.1 |
| Matplotlib | 3.10.1 |

Library Surprise tidak dipakai. Ketiga metode ditulis manual dengan NumPy/Pandas, supaya setiap
langkah bisa dijelaskan di laporan, sesuai permintaan tugas ("penjelasan dari langkah-langkah dalam
menjalankan metode").

Cara menjalankan ulang:

    python -m venv .venv
    .venv\Scripts\python -m pip install -r requirements.txt
    .venv\Scripts\python tests\test_metrics.py
    .venv\Scripts\python tests\test_ubcf.py
    .venv\Scripts\python tests\test_ibcf.py
    .venv\Scripts\python tests\test_mf.py
    .venv\Scripts\python tests\test_baseline.py
    .venv\Scripts\python scripts\mf_tuning.py     (opsional, ±25 menit, hasilnya sudah ada di results/)

---

## 2026-11-09 — Perbandingan akhir (notebook 05)

**Yang dilakukan:** ketiga metode dan empat baseline (`src/baseline.py`: tebak acak, rata-rata global,
rata-rata user, rata-rata film) dijalankan ulang di kelima fold. Hasil ketiga metode identik dengan notebook
02–04 (diperiksa otomatis di notebook).

**Alasan menambah baseline:** angka seperti MAE 0,73 atau NDCG 0,85 tidak bermakna tanpa patokan. Baseline
menunjukkan seberapa besar manfaat nyata dari metode yang lebih rumit.

**Tabel akhir (rata-rata ± simpangan baku 5 fold):**

| Metode | MAE | MSE | RMSE | NMSE | NDCG@10 |
|---|---|---|---|---|---|
| Tebak acak | 1,3902 ± 0,0167 | 2,8901 ± 0,0634 | 1,6999 ± 0,0186 | 0,7225 ± 0,0158 | 0,6839 ± 0,0275 |
| Rata-rata global | 0,9447 ± 0,0146 | 1,2672 ± 0,0394 | 1,1256 ± 0,0174 | 0,3168 ± 0,0099 | 0,7366 ± 0,0177 |
| Rata-rata user | 0,8362 ± 0,0090 | 1,0895 ± 0,0250 | 1,0437 ± 0,0119 | 0,2724 ± 0,0062 | 0,7366 ± 0,0177 |
| Rata-rata film | 0,8174 ± 0,0068 | 1,0498 ± 0,0146 | 1,0246 ± 0,0071 | 0,2624 ± 0,0036 | 0,8329 ± 0,0087 |
| User-based CF | **0,7342** ± 0,0047 | 0,8901 ± 0,0114 | 0,9434 ± 0,0060 | 0,2225 ± 0,0029 | 0,8474 ± 0,0088 |
| Item-based CF | 0,7383 ± 0,0086 | 0,8949 ± 0,0207 | 0,9459 ± 0,0109 | 0,2237 ± 0,0052 | 0,8375 ± 0,0064 |
| Matrix factorization | 0,7357 ± 0,0052 | **0,8554** ± 0,0097 | **0,9249** ± 0,0052 | **0,2138** ± 0,0024 | **0,8578** ± 0,0057 |

**Temuan utama:**

1. **Perbaikan terhadap baseline terkuat (rata-rata film):** MAE turun 9,68–10,18%, RMSE turun 7,67–9,73%,
   tetapi NDCG@10 hanya naik 0,55–2,99%.
2. **NDCG sebagian besar "gratis".** Mengurutkan film secara acak sudah memberi NDCG@10 = 0,684, dan
   mengurutkan berdasarkan rata-rata rating film memberi 0,833. Rata-rata 57,94% film milik setiap user di
   data uji bernilai relevan (≥ 4).
3. **Rata-rata global dan rata-rata user punya NDCG identik (0,7366)**, karena keduanya memberi skor sama untuk
   semua film milik seorang user, sehingga urutannya ditentukan oleh id film. Id film di MovieLens berkorelasi
   kuat dengan popularitas (Spearman −0,806), sehingga baseline ini tanpa sengaja mengurutkan film populer
   lebih dulu.
4. **Uji berpasangan per fold** (`results/uji_berpasangan.csv`): MF menang di kelima fold untuk RMSE (p = 0,0002
   vs UBCF; p = 0,0020 vs IBCF) dan NDCG@10 (p = 0,0102 dan 0,0008). Untuk MAE tidak ada perbedaan signifikan
   (p = 0,11–0,41). UBCF vs IBCF hanya berbeda signifikan di NDCG@10 (UBCF menang 5 fold, p = 0,0138).
   Catatan: dengan hanya 5 fold, uji ini kurang peka, jadi dibaca sebagai petunjuk.
5. **Sebaran kesalahan** menjelaskan mengapa MF unggul di RMSE tetapi tidak di MAE: MF paling jarang membuat
   kesalahan ≥ 2 bintang (3,23% vs 4,10% UBCF dan 4,29% IBCF), tetapi juga sedikit lebih jarang membuat
   tebakan yang sangat tepat (< 0,5 bintang: 41,19% vs 42,74% UBCF).
6. **Sparsity:** MAE untuk film dengan < 10 rating latih = 0,84–0,92, sedangkan untuk film dengan ≥ 150 rating
   = 0,70–0,71. IBCF paling terpengaruh pada film jarang (0,9216), tetapi sedikit paling baik pada film
   populer (0,7036).
7. **Waktu komputasi per fold:** UBCF 0,93 detik, IBCF 0,78 detik, MF 33,73 detik (hampir seluruhnya untuk
   pelatihan; prediksi 20.000 rating hanya 0,01 detik).
8. **Rekomendasi user 1:** MF dan UBCF berbagi 6 dari 10 film, UBCF dan IBCF berbagi 1 (Casablanca), MF dan
   IBCF tidak berbagi satu pun.

**Grafik baru:** `figures/perbandingan_metrik.png`, `figures/sebaran_error.png`, `figures/error_vs_popularitas.png`.

---

## 2026-11-09 — Dokumen masalah dan penyelesaian

**Yang dilakukan:** membuat [MASALAH_DAN_PENYELESAIAN.md](MASALAH_DAN_PENYELESAIAN.md), berisi 18 masalah (awalnya 15, ditambah 3 dari notebook 05)
yang ditemukan selama pengerjaan. Setiap masalah ditulis dengan pola masalah → bukti angka → penyebab →
penyelesaian → hasil, ditambah bagian persiapan tanya-jawab presentasi.

**Alasan:** permintaan kelompok agar kendala dan cara penyelesaiannya masuk ke laporan sebagai nilai tambah,
dan sebagai bekal jika diminta presentasi. Semua angka di dokumen tersebut sudah dicek ulang ke folder
`results/` dan notebook.

---

## Langkah berikutnya

1. Push notebook 05 dan dokumen masalah ke GitHub (hasil MF sudah di-push, commit 7c9cf6a).
2. Membagi penulisan bab laporan ke anggota kelompok (opsional: folder `laporan/` berisi template per bab,
   dikerjakan langsung di GitHub agar setiap anggota tercatat sebagai kontributor).
3. Laporan PDF, dengan bab khusus "Kendala dan Penyelesaian" dari MASALAH_DAN_PENYELESAIAN.md.
