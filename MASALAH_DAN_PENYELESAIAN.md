# Masalah yang Ditemukan dan Penyelesaiannya

Dokumen ini mengumpulkan 18 masalah yang muncul selama pengerjaan, beserta bukti angka, penyebab, dan cara
penyelesaiannya. Isinya disiapkan untuk:

1. **Bab "Kendala dan Penyelesaian"** di laporan PDF.
2. **Bahan presentasi**, jika kelompok diminta mempresentasikan hasil.

Setiap masalah ditulis dengan pola yang sama: **masalah → bukti → penyebab → penyelesaian → hasil**.
Semua angka berasal dari notebook dan folder `results/`, sehingga bisa ditunjukkan langsung saat ditanya.

---

## Ringkasan

| No | Masalah | Bagian | Penyelesaian | Dampak |
|---|---|---|---|---|
| 1 | Rumus NDCG di slide tidak konsisten | Metrik | Relevansi biner (rating ≥ 4) | Kedua versi rumus memberi hasil sama |
| 2 | NMSE punya beberapa versi di literatur | Metrik | Mengikuti versi slide | Hasil sesuai harapan dosen |
| 3 | Matriks rating sangat jarang terisi (93,7% kosong) | Data | Disadari sebagai tantangan utama; ditangani di masalah 4, 6, 10 | Menjelaskan banyak masalah lain |
| 4 | Pearson "tertipu" oleh user dengan sangat sedikit film yang sama | UBCF | Significance weighting | RMSE 0,9653 → 0,9536 |
| 5 | Rekomendasi didominasi film obscure | UBCF, IBCF, MF | Minimal 5 tetangga/rating pendukung | Daftar rekomendasi masuk akal |
| 6 | Cosine biasa menganggap hampir semua film mirip | IBCF | Adjusted cosine | RMSE 1,0508 → 0,9966 |
| 7 | Significance weighting malah memperburuk cosine biasa | IBCF | Dicatat sebagai temuan, tidak dipakai | Bahan diskusi |
| 8 | Angka di tabel slide M03 tidak cocok dengan hitungan | Slide | Dihitung ulang manual | Kode terbukti benar |
| 9 | 18 judul film terdaftar dua kali | Data | Penyaringan judul saat rekomendasi | Tidak ada judul ganda |
| 10 | Film di data uji yang tidak ada di data latih (cold start) | Semua | Cadangan: rata-rata rating user | Semua rating uji tetap terprediksi |
| 11 | Learning rate slide terlalu lambat untuk data besar | MF | Learning rate dipilih ulang | Pelatihan beberapa detik, bukan > 1 jam |
| 12 | MF cepat overfitting | MF | Regularisasi lebih kuat + early stopping | RMSE validasi 0,9732 → 0,9140 |
| 13 | Risiko memilih parameter dengan data uji | Metodologi | Data validasi terpisah / parameter dari slide | Hasil evaluasi jujur |
| 14 | Parameter terbaik berada di ujung rentang yang dicoba | MF | Rentang pencarian diperluas | Optimum yang sebenarnya ditemukan |
| 15 | Inisialisasi acak 0–1 tidak cocok untuk K besar | MF | Skala acak disesuaikan | Prediksi awal mendekati rata-rata rating |
| 16 | Angka metrik sulit dimaknai tanpa pembanding | Evaluasi | Empat baseline sederhana | Terlihat manfaat nyata tiap metode |
| 17 | NDCG terlihat tinggi, padahal sebagian besar "gratis" | Evaluasi | Dibandingkan dengan baseline acak dan rata-rata film | Klaim tentang NDCG tidak berlebihan |
| 18 | Selisih antar-metode sangat kecil, sulit menentukan pemenang | Evaluasi | Uji berpasangan per fold | Jelas mana perbedaan yang konsisten |

---

## Masalah pada Metrik Evaluasi

### 1. Rumus NDCG di slide tidak konsisten

**Masalah.** Di slide Evaluasi Part 2, DCG ditulis sebagai Σ rel_i / log₂(i+1), tetapi IDCG ditulis sebagai
Σ (2^rel_i − 1) / log₂(i+1). Jika dua rumus berbeda ini dicampur, NDCG bisa bernilai lebih dari 1, padahal
NDCG seharusnya berada di rentang 0–1.

**Penyelesaian.** Relevansi dibuat biner sesuai soal Kuis 2 di slide yang sama: film relevan (1) jika rating
aslinya ≥ 4, dan tidak relevan (0) jika di bawah 4. Dengan relevansi biner, kedua rumus memberi hasil
yang sama persis:

    jika rel = 1:  2¹ − 1 = 1  (sama dengan rel)
    jika rel = 0:  2⁰ − 1 = 0  (sama dengan rel)

Selain itu, contoh hitungan di slide sendiri (DCG₆ = 6,861, IDCG₆ = 7,141, nDCG₆ = 0,961) memakai rel_i
linear untuk keduanya. Fungsi NDCG kami diuji dengan contoh tersebut dan menghasilkan angka yang sama.

**Poin presentasi:** kelompok tidak hanya menyalin rumus, tetapi memeriksa konsistensinya dan memilih
pendekatan yang aman untuk kedua versi.

### 2. NMSE punya beberapa versi

**Masalah.** Di literatur, NMSE kadang dinormalisasi dengan variansi data, kadang dengan rentang rating.
Soal tugas meminta metrik "yang ada dalam slide".

**Penyelesaian.** Dipakai versi slide Evaluasi Part 1: NMSE = MSE / (r_max − r_min). Untuk MovieLens
(rating 1–5) pembaginya adalah 5 − 1 = 4. Slide juga punya salah ketik kecil ("rmax is the minimum
rating", maksudnya r_min), dan ini tidak memengaruhi perhitungan.

---

## Masalah pada Data

### 3. Matriks rating sangat jarang terisi (sparse)

**Bukti.** density = 100.000 / (943 × 1.682) = 100.000 / 1.586.126 ≈ 6,30%. Artinya **93,7%** pasangan
user–film tidak punya rating. Selain itu, 333 dari 1.682 film (19,8%) dirating kurang dari 5 kali.

**Dampak.** Sparsity inilah akar dari beberapa masalah berikutnya: dua user sering hanya punya sedikit
film yang sama (masalah 4), film jarang menghasilkan prediksi yang tidak bisa dipercaya (masalah 5), dan
ada film yang sama sekali tidak muncul di data latih (masalah 10). Ini sesuai dengan *sparsity problem*
yang dibahas di slide M03.

**Bukti dampaknya pada hasil (notebook 05).** Prediksi dikelompokkan berdasarkan jumlah rating film di data
latih:

| Jumlah rating film | User-based CF | Item-based CF | Matrix factorization |
|---|---|---|---|
| < 10 | 0,8415 | 0,9216 | 0,8464 |
| 10 – 49 | 0,7741 | 0,8044 | 0,7720 |
| 50 – 149 | 0,7325 | 0,7263 | 0,7307 |
| ≥ 150 | 0,7071 | 0,7036 | 0,7139 |

(angka = MAE). Semua metode makin buruk pada film yang jarang dirating. Item-based CF paling terpengaruh,
karena kemiripan sebuah film hanya bisa dihitung dari sedikit user. Menariknya, pada film populer item-based
CF justru sedikit paling baik. Grafik: `figures/error_vs_popularitas.png`.

### 9. Ada 18 judul film yang terdaftar dua kali

**Bukti.** Saat menampilkan rekomendasi item-based CF, film *Ulee's Gold (1997)* muncul dua kali.
Setelah diperiksa, MovieLens 100K memiliki 18 judul dengan dua id berbeda (misalnya *Chasing Amy* =
id 246 dan 268), mencakup 1.269 rating (1,27%).

**Penyelesaian.** Untuk evaluasi, id ganda dibiarkan terpisah, karena pembagian 5-fold resmi dibuat
berdasarkan id dan jumlahnya kecil. Untuk menampilkan rekomendasi, dibuat fungsi `unique_titles` yang
menyaring judul ganda dan judul yang sudah pernah dirating user melalui id kembarannya.

**Poin presentasi:** masalah ini hanya terlihat karena kami memeriksa hasil rekomendasi secara manual,
bukan hanya melihat angka metrik.

### 10. Film di data uji yang tidak ada di data latih (cold start)

**Bukti.** Di setiap fold ada 22–34 film di data uji yang tidak punya rating sama sekali di data latih.
Pada item-based CF, 29–38 rating uji per fold (sekitar 0,18%) tidak punya satu pun film mirip yang
pernah dirating user.

**Penyebab.** Ini *cold start problem* untuk item (slide M03): metode CF hanya bisa belajar dari rating
yang sudah ada.

**Penyelesaian.** Jika tidak ada informasi, prediksi memakai cadangan berupa rata-rata rating user
tersebut. Cara ini sama dengan *default back-off* di slide Evaluasi tambahan. Pada MF, vektor film yang
tidak pernah dilatih masih berisi angka acak, sehingga prediksinya juga diganti rata-rata user.

---

## Masalah pada User-Based CF

### 4. Pearson "tertipu" oleh user dengan sangat sedikit film yang sama

**Bukti.** Lima user yang paling mirip dengan user 1 menurut Pearson biasa semuanya punya korelasi
sempurna 1,0, tetapi hanya berdasarkan **2–3 film yang sama**. Setelah diperbaiki, tetangga terdekatnya
punya korelasi 0,49–0,58 dengan 49–69 film yang sama.

**Penyebab.** Dengan hanya 2 titik data, korelasi hampir selalu bernilai +1 atau −1. Analoginya, menganggap
seseorang punya selera musik yang sama dengan kita hanya karena kami sama-sama menyukai dua lagu.

**Penyelesaian.** *Significance weighting*, yang disebut di slide M02 bagian "Improving the metrics":

    sim'(a,b) = sim(a,b) × min(n_ab, 50) / 50

dengan n_ab = jumlah film yang dirating kedua user. Pasangan dengan ≥ 50 film sama tidak dikurangi.

**Hasil (fold 1, k = 30):**

| Varian | MAE | RMSE | NDCG@10 |
|---|---|---|---|
| Pearson biasa | 0,7552 | 0,9653 | 0,8336 |
| Pearson + significance weighting | **0,7416** | **0,9536** | **0,8446** |

Perbaikan terjadi di semua nilai k yang dicoba (10–100). Grafik: `figures/ubcf_sensitivitas_k.png`.

### 5. Rekomendasi didominasi film obscure

**Bukti.** Tanpa syarat tambahan, 10 rekomendasi teratas user 1 berisi film seperti *Santa with Muscles
(1996)* dan *Aiqing wansui (1994)* dengan prediksi rating sempurna 5,0, tetapi hanya didukung oleh
**1–3 tetangga**.

**Penyebab.** Jika hanya satu tetangga yang menonton sebuah film dan memberinya rating tinggi, prediksinya
otomatis tinggi. Bukti satu orang terlalu lemah, tetapi rumus prediksi tidak membedakannya.

**Penyelesaian.** Saat membuat daftar rekomendasi, film harus didukung minimal 5 tetangga (UBCF), 5 film
mirip (IBCF), atau 5 rating di data latih (MF). Aturan ini hanya dipakai untuk daftar rekomendasi, tidak
untuk evaluasi metrik, supaya angka MAE/RMSE tetap mencerminkan semua prediksi.

**Hasil.** Rekomendasi UBCF untuk user 1 berubah menjadi *Casablanca*, *Schindler's List*, *Rear Window*,
*Lawrence of Arabia*, dan film lain yang dikenal luas. Keempat film tersebut dirating oleh 173–298 user
dengan rata-rata rating 4,23–4,47, jauh di atas rata-rata seluruh dataset (3,53).

---

## Masalah pada Item-Based CF

### 6. Cosine biasa menganggap hampir semua film mirip

**Bukti.** Lima film yang paling mirip dengan *Star Wars (1977)* menurut cosine biasa bernilai
0,979–0,989. Nilai sedekat ini hampir tidak membedakan film yang benar-benar mirip dari yang tidak.

**Penyebab.** Semua rating bernilai positif (1–5), sehingga vektor rating dua film hampir selalu searah
(sudutnya kecil, cosine mendekati 1). Cosine biasa juga mengabaikan kebiasaan user: rating 3 dari user
yang biasanya memberi 5 sebenarnya berarti "kurang suka".

**Penyelesaian.** *Adjusted cosine similarity* (slide M03): setiap rating dikurangi rata-rata rating
user tersebut sebelum dihitung. Nilainya menjadi lebih tersebar (0,63–0,82 untuk contoh Star Wars) dan
bisa negatif.

**Hasil (fold 1, k = 30):**

| Varian | MAE | RMSE | NDCG@10 |
|---|---|---|---|
| Cosine biasa | 0,8283 | 1,0508 | 0,7510 |
| Adjusted cosine | 0,7851 | 0,9966 | 0,8210 |
| Adjusted cosine + significance weighting | **0,7523** | **0,9644** | **0,8308** |

Grafik: `figures/ibcf_sensitivitas_k.png`.

### 7. Significance weighting malah memperburuk cosine biasa

**Bukti.** Pada cosine biasa, significance weighting menaikkan RMSE dari 1,0508 ke 1,1003, padahal pada
adjusted cosine perbaikan yang sama menurunkan RMSE.

**Dugaan penyebab.** Karena nilai cosine biasa hampir sama semua (≈ 0,98), setelah dikalikan bobot jumlah
user, urutan tetangga praktis hanya ditentukan oleh popularitas film, bukan kemiripan selera. Ini masih
dugaan dan belum dibuktikan dengan eksperimen terpisah.

**Keputusan.** Cosine biasa tidak dipakai di model akhir. Temuan ini disimpan sebagai bahan diskusi:
teknik perbaikan yang membantu di satu kondisi belum tentu membantu di kondisi lain.

### 8. Angka di tabel slide M03 tidak cocok dengan hitungan

**Bukti.** Tabel similarity film di slide menulis Forrest Gump vs The GodFather = 0,8. Dihitung manual
dari user yang merating keduanya (Robert: 3 dan 1, David: 2 dan 2):

    cos = (3·1 + 2·2) / (√(3² + 2²) · √(1² + 2²)) = 7 / (√13 · √5) = 7 / 8,062 = 0,868

**Penyelesaian.** Kode menghasilkan 0,868, sama dengan hitungan manual. Angka lain di contoh yang sama
(0,792 dan 0,902) cocok dengan slide, sehingga kemungkinan besar angka 0,8 di slide adalah salah ketik
atau pembulatan.

**Poin presentasi:** setiap rumus kami uji dengan contoh di slide (`tests/`), sehingga ketidakcocokan
seperti ini bisa terdeteksi.

---

## Masalah pada Matrix Factorization

### 11. Learning rate dari slide terlalu lambat untuk data besar

**Bukti.** Kode slide memakai α = 0,0002 dan 5.000 iterasi untuk matriks 5 × 4 (12 rating). Pada
MovieLens, satu epoch (80.000 rating) butuh sekitar 0,8 detik, sehingga 5.000 epoch akan memakan lebih
dari 1 jam per fold. Dengan α = 0,0002, RMSE validasi masih terus turun sampai epoch 60 (0,9647).

**Penyelesaian.** Rumus update tetap sama dengan slide, tetapi nilai α dipilih ulang. Dengan α = 0,01
dan regularisasi yang tepat, model terbaik dicapai dalam 35 epoch (sekitar 30 detik).

Grafik: `figures/mf_learning_rate.png`.

### 12. MF cepat overfitting

**Bukti.** Dengan β = 0,02 (nilai slide) dan α = 0,01, RMSE validasi mencapai titik terbaik di epoch 3
(0,9732), lalu **naik terus** sampai 1,0976 di epoch 60, sementara RMSE data latih terus turun.

**Penyebab.** *Overfitting*: angka-angka di matriks P dan Q membesar demi menghafal data latih, sehingga
model makin buruk menebak rating yang belum pernah dilihat. Regularisasi β = 0,02 terlalu lemah untuk
menahannya, terutama dengan banyak fitur (K besar).

**Penyelesaian.**
1. Regularisasi diperkuat menjadi β = 0,2 (dipilih dengan data validasi).
2. *Early stopping*: pelatihan dihentikan di epoch dengan RMSE validasi terkecil (epoch 35).

**Hasil.** RMSE validasi terbaik turun dari 0,9732 menjadi 0,9140, dan kurva validasi tidak naik lagi
(`figures/mf_kurva_belajar.png`). Temuan menarik: K besar baru bermanfaat jika regularisasinya cukup
kuat. Dengan β = 0,02, K = 50 justru lebih buruk daripada K = 5.

### 14. Parameter terbaik berada di ujung rentang yang dicoba

**Masalah.** Pada pencarian awal (K = 5–50, β = 0,02–0,1), hasil terbaik adalah K = 50 dan β = 0,1, yaitu
nilai **terbesar** di kedua rentang. Artinya bisa jadi nilai yang lebih baik berada di luar rentang.

**Penyelesaian.** Rentang pencarian diperluas dua kali (β sampai 0,5, K sampai 200, epoch sampai 100).
Pencarian dihentikan setelah nilai terbaik (K = 100, β = 0,2) diapit nilai yang lebih buruk di kedua sisi:

| | β = 0,1 | β = 0,2 | β = 0,3 |
|---|---|---|---|
| K = 50 | 0,9413 | 0,9185 | 0,9249 |
| K = 100 | 0,9274 | **0,9140** | 0,9257 |
| K = 200 | | 0,9157 | 0,9262 |

### 15. Inisialisasi acak 0–1 tidak cocok untuk K besar

**Masalah.** Kode slide mengisi P dan Q dengan angka acak 0–1. Rata-rata hasil kali dua angka acak 0–1
adalah 0,25, sehingga prediksi awal p · q ≈ K × 0,25. Untuk K = 100, prediksi awal sekitar 25, jauh di
atas rating maksimum 5.

**Penyelesaian.** Angka acak diambil dari 0 sampai s, dengan s dipilih agar prediksi awal kira-kira sama
dengan rata-rata rating (3,53):

    K × (s/2)² = rata-rata rating   →   s = 2 × √(rata-rata rating / K)

Contoh slide (K = 2) tetap memakai acak 0–1 seperti aslinya.

---

## Masalah Metodologi

### 13. Risiko memilih parameter dengan data uji

**Masalah.** Jika k, β, atau jumlah epoch dipilih berdasarkan nilai terbaik di data uji, hasil evaluasi
akan terlihat lebih bagus dari kenyataan, karena data uji sudah "diintip". Ini disebut *data leakage*.

**Penyelesaian.**
- **UBCF dan IBCF:** k = 30 dan γ = 50 ditetapkan **sebelum** melihat hasil, berdasarkan slide M02
  (20–50 tetangga wajar untuk MovieLens) dan Sarwar et al. (2001). Tabel perbandingan k di notebook
  hanya analisis sensitivitas, bukan alat memilih k.
- **MF:** karena parameternya harus dipilih ulang (masalah 11–12), data latih fold 1 dibagi lagi menjadi
  90% latih dan 10% **validasi**. Semua pemilihan memakai data validasi; data uji baru dipakai di akhir.

**Poin presentasi:** ini menunjukkan bahwa kelompok memahami cara evaluasi yang benar, bukan hanya
mengejar angka terbaik.

### 16. Angka metrik sulit dimaknai tanpa pembanding

**Masalah.** MAE 0,73 atau NDCG 0,85 tidak bisa disebut "bagus" atau "buruk" tanpa patokan.

**Penyelesaian.** Ditambahkan empat baseline yang sangat sederhana: tebak acak, rata-rata global, rata-rata
user, dan rata-rata film (`src/baseline.py`). Metode yang lebih rumit baru bisa disebut berguna jika jelas
mengalahkan baseline ini.

**Hasil.** Dibanding baseline terkuat (rata-rata film, MAE 0,8174 dan RMSE 1,0246), ketiga metode menurunkan
MAE 9,68–10,18% dan RMSE 7,67–9,73%. Contoh perhitungan untuk RMSE matrix factorization:

    perbaikan = (1,0246 − 0,9249) / 1,0246 × 100% = 9,73%

### 17. NDCG terlihat tinggi, padahal sebagian besar "gratis"

**Bukti.** NDCG@10 ketiga metode 0,84–0,86, tetapi:

| Cara mengurutkan film | NDCG@10 |
|---|---|
| Acak | 0,6839 |
| Berdasarkan rata-rata rating film | 0,8329 |
| Matrix factorization (terbaik) | 0,8578 |

**Penyebab.** NDCG dihitung dengan mengurutkan film-film milik user di data uji, dan rata-rata 57,94% film
tersebut relevan (rating ≥ 4). Jadi urutan acak pun sudah banyak menempatkan film relevan di posisi atas.

**Penyelesaian.** Nilai NDCG selalu dilaporkan berdampingan dengan baseline. Kesimpulan yang ditarik adalah
perbandingan antar-metode (MF > UBCF > IBCF untuk NDCG), bukan klaim bahwa 0,86 adalah nilai yang tinggi.

**Temuan sampingan.** Baseline rata-rata global dan rata-rata user punya NDCG identik (0,7366) dan lebih tinggi
dari acak. Keduanya memberi skor sama untuk semua film milik seorang user, sehingga urutannya ditentukan oleh
id film. Ternyata id film di MovieLens berkorelasi kuat dengan popularitas (korelasi Spearman −0,806), sehingga
kedua baseline ini tanpa sengaja mengurutkan film populer lebih dulu.

### 18. Selisih antar-metode sangat kecil, sulit menentukan pemenang

**Masalah.** Selisih MAE ketiga metode hanya 0,0015–0,0041. Apakah selisih sekecil ini berarti, atau hanya
kebetulan?

**Penyelesaian.** Karena semua metode diuji pada 5 fold yang sama, dibuat perbandingan berpasangan: di setiap
fold, metode mana yang menang, ditambah uji t berpasangan (`results/uji_berpasangan.csv`).

**Hasil.**

| Perbandingan | MAE | RMSE | NDCG@10 |
|---|---|---|---|
| MF vs UBCF | 2 : 3 fold, p = 0,41 | **5 : 0, p = 0,0002** | **5 : 0, p = 0,010** |
| MF vs IBCF | 4 : 1 fold, p = 0,27 | **5 : 0, p = 0,0020** | **5 : 0, p = 0,0008** |
| UBCF vs IBCF | 4 : 1 fold, p = 0,11 | 3 : 2, p = 0,38 | **5 : 0, p = 0,014** |

Keunggulan MF di RMSE dan NDCG konsisten di semua fold, sedangkan perbedaan MAE tidak bisa dibedakan dari
kebetulan. Catatan: dengan hanya 5 fold, uji ini kurang peka, sehingga dibaca sebagai petunjuk.

Sebaran kesalahan menjelaskan mengapa MF unggul di RMSE tetapi tidak di MAE: MF paling jarang membuat
kesalahan ≥ 2 bintang (3,23%, dibanding 4,10% UBCF dan 4,29% IBCF), tetapi juga sedikit lebih jarang menebak
sangat tepat (< 0,5 bintang: 41,19% vs 42,74% UBCF). Grafik: `figures/sebaran_error.png`.

---

## Persiapan Tanya-Jawab Presentasi

**Mengapa memilih MovieLens 100K?**
Memenuhi syarat ada *user history* (user, film, rating, timestamp; setiap user minimal 20 rating),
ukurannya pas untuk implementasi manual, sudah menyediakan pembagian 5-fold (sesuai materi 5-fold cross
validation di slide), dan dipakai sebagai contoh di slide M02.

**Bagaimana kalian yakin kodenya benar?**
Setiap rumus diuji dengan contoh hitungan di slide: MAE 1,75 (Evaluasi tambahan), nDCG 0,961 (Evaluasi
Part 2), Pearson 0,85 / 0,70 / 0,00 / −0,79 dan prediksi Alice 4,87 (M02), cosine 0,792 dan adjusted
cosine 0,80 (M03), serta rekonstruksi matriks contoh M04. Semua test ada di folder `tests/`.

**Metode mana yang terbaik?**
Matrix factorization unggul di MSE, RMSE, NMSE, dan NDCG@10, dan keunggulan di RMSE serta NDCG@10 konsisten
di kelima fold (uji t berpasangan, p < 0,05). Untuk MAE, ketiga metode praktis setara: selisihnya hanya
0,0015–0,0041 dan tidak signifikan. Sebabnya, MF lebih jarang membuat kesalahan besar (≥ 2 bintang: 3,23%
vs 4,10% UBCF), dan RMSE sangat menghukum kesalahan besar. Perbedaan UBCF dan IBCF sangat kecil, sejalan
dengan slide M02 yang menyebut perbedaan antar metode CF sering hanya sekitar 1/100. Lihat masalah 18.

**Mengapa nilai NDCG kalian tinggi (sekitar 0,85)?**
Karena cara hitungnya, bukan karena modelnya luar biasa. NDCG dihitung dengan mengurutkan film-film milik
user di data uji, dan rata-rata 57,94% film tersebut relevan (rating ≥ 4). Urutan acak saja sudah memberi
NDCG@10 = 0,684, dan mengurutkan berdasarkan rata-rata rating film memberi 0,833. Jadi angka NDCG kami
gunakan untuk **membandingkan** metode, bukan sebagai klaim bahwa nilainya tinggi. Lihat masalah 17.

**Apakah metode kalian benar-benar lebih baik dari cara sederhana?**
Ya, untuk prediksi rating. Dibanding baseline terkuat (rata-rata rating film), ketiga metode menurunkan MAE
sekitar 10% dan RMSE 7,7–9,7%. Untuk urutan rekomendasi (NDCG), perbaikannya kecil (0,5–3,0%). Lihat
masalah 16 dan 17.

**Mengapa tidak memakai library seperti Surprise?**
Tugas meminta penjelasan langkah-langkah metode. Dengan menulis sendiri, setiap langkah bisa dijelaskan,
diuji dengan contoh slide, dan dimodifikasi (misalnya menambah significance weighting atau mengganti
cara inisialisasi MF), sesuatu yang lebih sulit dilakukan jika memakai library siap pakai.

**Mengapa dataset tidak ada di repository?**
Lisensi MovieLens melarang penyebaran ulang dataset tanpa izin ("The user may not redistribute the data
without separate permission"). Karena itu dataset tidak diunggah. Sebagai gantinya, kode mengunduh dataset
otomatis dari situs resmi GroupLens saat pertama kali dijalankan dan mencocokkannya dengan SHA-256 file resmi,
sehingga siapa pun yang menjalankan kode memakai data yang sama persis. Lisensi yang sama juga mewajibkan
sitasi Harper & Konstan (2015), yang kami cantumkan di daftar pustaka.

**Mengapa memakai Jupyter Notebook?**
Soal tugas meminta penjelasan langkah-langkah metode beserta screenshot source code yang dijelaskan.
Notebook menampilkan tiga hal dalam satu tempat: penjelasan (teks dan rumus), kode, dan hasilnya tepat di
bawah kode tersebut, sehingga satu screenshot sudah memuat kode beserta hasilnya. Format ini juga sejalan
dengan slide M04 yang meminta kode MF dijalankan di Google Colab (Colab adalah Jupyter Notebook versi
online). Selain itu, notebook di repository sudah berisi output dan grafik, sehingga bisa dibaca langsung di
GitHub tanpa dijalankan ulang.

Kelemahan notebook juga diantisipasi:

| Kelemahan notebook | Cara mengatasinya |
|---|---|
| Kode sulit dipakai ulang antar-notebook | Rumus inti ditulis di `src/` dan dipanggil oleh notebook |
| Sulit diuji otomatis | Pengujian terpisah di `tests/` |
| Hasil bisa tidak konsisten jika sel dijalankan tidak berurutan | Semua notebook dijalankan ulang dari sel pertama sampai terakhir sebelum disimpan |
| Proses berat membuat notebook lambat | Tuning MF (±25 menit) dipisah menjadi `scripts/mf_tuning.py` |

Singkatnya, folder `src/` adalah "mesin" (rumus yang bisa diuji dan dipakai ulang), sedangkan notebook
adalah "etalase" yang menjelaskan langkah dan hasilnya.

**Mengapa parameter MF berbeda dari slide?**
Lihat masalah 11 dan 12. Rumus update-nya tetap sama dengan slide; yang berubah hanya nilai α, β, K, dan
jumlah epoch, karena nilai slide dirancang untuk matriks 5 × 4, bukan 80.000 rating.

**Apa kelemahan sistem kalian?**
Belum bisa menangani user atau film baru tanpa rating (*cold start*), hanya memakai rating (tidak memakai
genre atau data user), data MovieLens sudah lama (1997–1998), dan NDCG hanya mengurutkan film yang ada di
data uji, bukan seluruh katalog film.
