"""Menjalankan percobaan 5-fold untuk satu metode, dengan cache.

Notebook laporan hanya memanggil fungsi di sini; rumus model tetap di
ubcf.py, ibcf.py, mf.py, baseline.py dan rumus metrik di metrics.py.
"""
import re
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import cache
from .data import load_fold, load_ratings, to_matrix
from .metrics import evaluate

KOLOM_PRED = ["user", "item", "rating", "pred", "rating_film_latih"]


def _slug(teks):
    return re.sub(r"[^a-z0-9]+", "_", teks.lower()).strip("_")


def _params(cls, kwargs, fold):
    return {"kelas": cls.__name__, "kwargs": kwargs, "fold": fold}


def jalankan_fold(cls, kwargs, fold):
    """Melatih satu model di satu fold, memprediksi data uji, dan menghitung kelima metrik.

    Waktu latih dan waktu prediksi dicatat terpisah dengan time.perf_counter().
    """
    train, test = load_fold(fold)
    R = to_matrix(train)
    n_rating_film = (~np.isnan(R)).sum(axis=0)

    t0 = time.perf_counter()
    model = cls(**kwargs).fit(R)
    t1 = time.perf_counter()
    pred = model.predict_df(test)
    t2 = time.perf_counter()

    pred["rating_film_latih"] = n_rating_film[test["item"].to_numpy() - 1]
    extra = [c for c in ("n_tetangga",) if c in pred.columns]
    return {
        "metrik": evaluate(pred),
        "latih_detik": t1 - t0,
        "prediksi_detik": t2 - t1,
        "pred": pred[KOLOM_PRED + extra].reset_index(drop=True),
    }


def jalankan_cv(nama, cls, kwargs, folds=range(1, 6), n_jobs=1, force=False):
    """Menjalankan metode di semua fold. Fold yang sudah ada di cache tidak dihitung ulang.

    n_jobs > 1 menjalankan fold yang belum ada di cache secara paralel
    (dipakai untuk matrix factorization yang lambat; waktu latih per fold
    tetap diukur di dalam prosesnya masing-masing).

    Mengembalikan (hasil, waktu, prediksi, status):
      hasil    : DataFrame Metode, Fold, MAE, MSE, RMSE, NMSE, NDCG@10
      waktu    : DataFrame Metode, Fold, Latih (detik), Prediksi (detik)
      prediksi : DataFrame prediksi data uji semua fold (kolom Metode, Fold, user, item, ...)
      status   : "cache" jika semua fold dimuat dari cache, selain itu "dihitung N fold"
    """
    folds = list(folds)
    cache_nama = {k: f"{_slug(nama)}_fold{k}" for k in folds}
    out = {k: (None if force else cache.load(cache_nama[k], _params(cls, kwargs, k))) for k in folds}
    kurang = [k for k in folds if out[k] is None]

    if kurang:
        if n_jobs > 1 and len(kurang) > 1:
            baru = Parallel(n_jobs=min(n_jobs, len(kurang)))(
                delayed(jalankan_fold)(cls, kwargs, k) for k in kurang)
        else:
            baru = [jalankan_fold(cls, kwargs, k) for k in kurang]
        for k, nilai in zip(kurang, baru):
            cache.store(cache_nama[k], _params(cls, kwargs, k), nilai)
            out[k] = nilai

    hasil = pd.DataFrame([{"Metode": nama, "Fold": k, **out[k]["metrik"]} for k in folds])
    waktu = pd.DataFrame([{"Metode": nama, "Fold": k, "Latih (detik)": out[k]["latih_detik"],
                           "Prediksi (detik)": out[k]["prediksi_detik"]} for k in folds])
    prediksi = pd.concat([out[k]["pred"].assign(Metode=nama, Fold=k) for k in folds], ignore_index=True)
    status = "cache" if not kurang else f"dihitung ({len(kurang)} dari {len(folds)} fold)"
    return hasil, waktu, prediksi, status


def rekomendasi_user(nama, cls, kwargs, fungsi_rekomendasi, user=0, n=40, force=False):
    """Melatih model dengan SELURUH 100.000 rating, lalu merekomendasikan film untuk satu user.

    fungsi_rekomendasi(model, R_penuh, user, n) -> daftar (indeks_film, prediksi, jumlah_pendukung).
    Yang disimpan di cache hanya daftar rekomendasinya (bukan model) agar kecil.
    """
    params = {"kelas": cls.__name__, "kwargs": kwargs, "user": user, "n": n, "data": "penuh"}

    def hitung():
        R_penuh = to_matrix(load_ratings())
        t0 = time.perf_counter()
        model = cls(**kwargs).fit(R_penuh)
        latih = time.perf_counter() - t0
        return {"rek": fungsi_rekomendasi(model, R_penuh, user, n), "latih_detik": latih}

    nilai, dari_cache = cache.cached(f"{_slug(nama)}_rekomendasi_user{user + 1}", params, hitung)
    return nilai["rek"], dari_cache


def ringkas_fold(hasil):
    """Tabel per fold ditambah baris Rata-rata dan Simpangan baku (indeks = Fold)."""
    t = hasil.drop(columns="Metode").set_index("Fold")
    n = len(t)
    t.loc["Rata-rata"] = t.mean()
    t.loc["Simpangan baku"] = t.iloc[:n].std()
    return t


def sensitivitas(cls, varian, ks, fold=1, force=False):
    """Pengaruh jumlah tetangga k (dan varian lain) terhadap metrik di satu fold.

    varian : {label: kwargs tanpa k}. Setiap varian dilatih SEKALI, lalu atribut k
    diganti-ganti (similarity tidak bergantung pada k, jadi tidak perlu dilatih ulang).
    Ini hanya analisis sensitivitas; konfigurasi akhir tidak dipilih dari sini.
    """
    params = {"kelas": cls.__name__, "varian": varian, "ks": list(ks), "fold": fold}

    def hitung():
        train, test = load_fold(fold)
        R = to_matrix(train)
        baris = []
        for label, kwargs in varian.items():
            m = cls(**kwargs).fit(R)
            for k in ks:
                m.k = k
                baris.append({"Varian": label, "k": k, **evaluate(m.predict_df(test))})
        return pd.DataFrame(baris)

    tabel, dari_cache = cache.cached(f"sensitivitas_{_slug(cls.__name__)}_fold{fold}", params, hitung, force=force)
    return tabel, dari_cache
