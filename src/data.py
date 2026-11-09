"""Memuat dataset MovieLens 100K dan mengubahnya menjadi matriks rating user x item."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ml-100k"
N_USERS = 943
N_ITEMS = 1682
R_MIN, R_MAX = 1, 5

COLS = ["user", "item", "rating", "timestamp"]


def load_ratings(filename="u.data"):
    """Membaca file rating (dipisah tab): user, item, rating, timestamp."""
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
    text = (DATA_DIR / "u.item").read_bytes().decode("latin-1")
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
