"""Cache hasil komputasi berat ke results/cache/*.pkl.

Tujuan: "Run All" notebook laporan cukup beberapa detik jika hasilnya sudah
pernah dihitung. Nama berkas cache berisi tiga hal, sehingga cache otomatis
dianggap usang (dihitung ulang) jika salah satunya berubah:

    <nama>_<hash parameter>_<hash kode>.pkl

- hash parameter : hyperparameter dan fold yang dipakai
- hash kode      : isi berkas src/ yang memengaruhi hasil (data, metrik, model)

Untuk memaksa hitung ulang tanpa menunggu salah satunya berubah, beri
force=True (di notebook: FORCE_RECOMPUTE = True).
"""
import hashlib
import json
import pickle
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "results" / "cache"

# Modul yang memengaruhi angka hasil. plot_style/report/cache sengaja tidak ikut,
# supaya mengubah tampilan grafik tidak menghapus cache.
KODE_PENENTU = ["data", "metrics", "baseline", "ubcf", "ibcf", "mf", "experiment"]


@lru_cache(maxsize=1)
def code_fingerprint():
    """Sidik jari isi berkas kode penentu hasil."""
    h = hashlib.sha256()
    for nama in KODE_PENENTU:
        h.update(nama.encode())
        h.update((ROOT / "src" / f"{nama}.py").read_bytes())
    return h.hexdigest()[:10]


def _path(name, params):
    kunci = hashlib.sha256(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:10]
    return CACHE_DIR / f"{name}_{kunci}_{code_fingerprint()}.pkl", f"{name}_{kunci}_"


def load(name, params):
    """Mengembalikan nilai tersimpan, atau None jika belum ada / sudah usang."""
    path, _ = _path(name, params)
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def store(name, params, value):
    """Menyimpan nilai, dan membuang cache lama dengan nama + parameter yang sama."""
    path, prefix = _path(name, params)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for lama in CACHE_DIR.glob(prefix + "*.pkl"):
        lama.unlink()
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(value, f)
    tmp.replace(path)


def cached(name, params, compute, force=False):
    """Memuat dari cache jika ada, jika tidak hitung dengan compute() lalu simpan.

    Mengembalikan (nilai, dari_cache) agar notebook bisa menampilkan statusnya.
    """
    if not force:
        nilai = load(name, params)
        if nilai is not None:
            return nilai, True
    nilai = compute()
    store(name, params, nilai)
    return nilai, False
