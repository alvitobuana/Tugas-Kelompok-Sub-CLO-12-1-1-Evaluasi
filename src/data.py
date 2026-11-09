"""Memuat dataset MovieLens 100K dan mengubahnya menjadi matriks rating user x item.

Dataset tidak disimpan di repository karena lisensi MovieLens melarang
penyebaran ulang ("The user may not redistribute the data without separate
permission"). Jika folder data/ml-100k belum ada, dataset diunduh otomatis
dari situs resmi GroupLens saat pertama kali dibutuhkan.
"""
import hashlib
import ssl
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DATA = Path(__file__).resolve().parent.parent / "data"
DATA_DIR = ROOT_DATA / "ml-100k"
URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
SHA256 = "50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229"
N_USERS = 943
N_ITEMS = 1682
R_MIN, R_MAX = 1, 5

COLS = ["user", "item", "rating", "timestamp"]


def _unduh(url):
    """Mengunduh url dengan verifikasi sertifikat TLS tetap aktif.

    Python di Windows kadang gagal memverifikasi rantai sertifikat
    (CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate) karena
    tidak mengambil sertifikat perantara sendiri. Jika itu terjadi, dicoba lagi
    memakai penyimpanan sertifikat sistem operasi lewat paket `truststore`.
    Verifikasi tidak pernah dimatikan.
    """
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            return resp.read()
    except urllib.error.URLError as err:
        if not isinstance(err.reason, ssl.SSLCertVerificationError):
            raise
        try:
            import truststore
        except ImportError:
            raise RuntimeError(
                "Verifikasi sertifikat gagal. Pasang `pip install truststore` lalu jalankan ulang, "
                f"atau unduh manual {url} dan ekstrak ke folder data/ (lihat README)."
            ) from err
        ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        with urllib.request.urlopen(url, timeout=60, context=ctx) as resp:
            return resp.read()


def ensure_data():
    """Mengunduh dan mengekstrak MovieLens 100K jika belum ada.

    Isi file yang diunduh dicocokkan dengan SHA-256 file resmi, supaya yang
    dipakai pasti dataset yang sama dengan yang kami gunakan.
    """
    if (DATA_DIR / "u.data").exists():
        return
    ROOT_DATA.mkdir(exist_ok=True)
    zip_path = ROOT_DATA / "ml-100k.zip"
    if not zip_path.exists():
        print(f"Dataset belum ada, mengunduh dari {URL} ...")
        zip_path.write_bytes(_unduh(URL))
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    if digest != SHA256:
        zip_path.unlink()
        raise RuntimeError("File unduhan tidak sesuai dengan dataset resmi (SHA-256 berbeda). Coba jalankan ulang.")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(ROOT_DATA)
    print(f"Dataset siap di {DATA_DIR}")


def verifikasi_dataset():
    """Mengembalikan (sha256_zip, cocok). Keduanya None jika zip tidak ada (dataset diekstrak manual)."""
    zip_path = ROOT_DATA / "ml-100k.zip"
    if not zip_path.exists():
        return None, None
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    return digest, digest == SHA256


def load_ratings(filename="u.data"):
    """Membaca file rating (dipisah tab): user, item, rating, timestamp."""
    ensure_data()
    return pd.read_csv(DATA_DIR / filename, sep="\t", names=COLS)


def load_fold(k):
    """Mengambil pembagian train/test bawaan MovieLens untuk fold ke-k (1..5)."""
    train = load_ratings(f"u{k}.base")
    test = load_ratings(f"u{k}.test")
    return train, test


def load_items():
    """Judul film, diindeks dengan item id.

    File u.item memakai encoding latin-1 (misalnya huruf é pada "Véronique"),
    bukan UTF-8, sehingga harus di-decode sebagai latin-1.
    """
    ensure_data()
    text =(DATA_DIR / "u.item").read_bytes().decode("latin-1")
    rows = [line.split("|")[:2] for line in text.splitlines() if line]
    return pd.Series({int(i): t for i, t in rows}, name="title").rename_axis("item")


def unique_titles(recs, titles, seen_items, n=10):
    """Menyaring daftar rekomendasi (indeks 0-based) agar tidak ada judul ganda.

    MovieLens 100K punya 18 judul yang terdaftar dua kali dengan id berbeda.
    Film yang judulnya sudah pernah dirating user (lewat id kembarannya) atau
    sudah ada di daftar rekomendasi dilewati.
    """
    seen = {titles[i + 1] for i in seen_items}
    out = []
    for rec in recs:
        title = titles[rec[0] + 1]
        if title not in seen:
            seen.add(title)
            out.append(rec)
    return out[:n]


def to_matrix(df, n_users=N_USERS, n_items=N_ITEMS):
    """Mengubah tabel rating menjadi matriks R (user x item).

    Sel yang tidak punya rating diisi NaN (bukan 0), karena 0 bukan rating
    yang valid, melainkan 'belum pernah dirating'. Id user/item di MovieLens
    mulai dari 1, sedangkan indeks NumPy mulai dari 0, jadi dikurangi 1.
    """
    R = np.full((n_users, n_items), np.nan)
    R[df["user"].to_numpy() - 1, df["item"].to_numpy() - 1] = df["rating"].to_numpy()
    return R
