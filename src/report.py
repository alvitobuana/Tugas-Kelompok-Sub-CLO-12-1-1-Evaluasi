"""Pembantu tampilan untuk notebook laporan: tabel, gambar, dan keterangannya.

Semua tabel dan gambar lewat sini supaya:
- penomoran Gambar/Tabel otomatis berurutan dan konsisten dengan narasi,
- tabel tampil sebagai Markdown (rapi di notebook, dan diubah pandoc menjadi
  tabel LaTeX asli saat export),
- angka desimal memakai koma (kebiasaan penulisan Indonesia), sama seperti narasi,
- gambar tersimpan di figures/fig_<bagian>_<nama>.png dengan dpi 150.
"""
from pathlib import Path

import pandas as pd
from IPython.display import Markdown, display

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"


def angka(x, desimal=4):
    """Format angka gaya Indonesia: titik ribuan dan koma desimal, contoh 1234.5 -> '1.234,5000'."""
    return f"{x:,.{desimal}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def persen(x, desimal=2):
    """Format persentase gaya Indonesia, contoh 0.553 -> '55,30%'."""
    return angka(x * 100, desimal) + "%"


def _tabel_markdown(df, floatfmt, index):
    """DataFrame -> tabel Markdown dengan koma desimal dan kolom angka rata kanan."""
    isi = df.copy()
    angka_kolom = []
    for c in isi.columns:
        if pd.api.types.is_float_dtype(isi[c]):
            isi[c] = isi[c].map(lambda v: "" if pd.isna(v) else f"{v:{floatfmt}}".replace(".", ","))
            angka_kolom.append(True)
        else:
            angka_kolom.append(pd.api.types.is_integer_dtype(isi[c]))
    rata = (["left"] if index else []) + ["right" if a else "left" for a in angka_kolom]
    return isi.to_markdown(index=index, colalign=rata, disable_numparse=True)


class Laporan:
    """Menghitung nomor Gambar dan Tabel, dan mencatatnya agar bisa dirujuk di narasi."""

    def __init__(self):
        self.n_gambar = 0
        self.n_tabel = 0
        self.nomor = {}          # kunci -> "Gambar 3" / "Tabel 5"
        FIG_DIR.mkdir(exist_ok=True)

    @staticmethod
    def md(teks):
        """Menampilkan teks sebagai Markdown (dipakai untuk narasi yang berisi angka hasil)."""
        display(Markdown(teks))

    def tabel(self, df, keterangan, kunci=None, floatfmt=".4f", index=True):
        """Menampilkan DataFrame sebagai tabel Markdown bernomor.

        Keterangan tabel diletakkan di atas tabel (kebiasaan penulisan ilmiah).
        """
        self.n_tabel += 1
        label = f"Tabel {self.n_tabel}"
        if kunci:
            self.nomor[kunci] = label
        display(Markdown(f"**{label}.** {keterangan}\n\n{_tabel_markdown(df, floatfmt, index)}"))
        return label

    def gambar(self, fig, bagian, nama, keterangan, kunci=None):
        """Menyimpan gambar ke figures/ lalu menampilkan keterangannya di bawah gambar."""
        import matplotlib.pyplot as plt

        self.n_gambar += 1
        label = f"Gambar {self.n_gambar}"
        if kunci:
            self.nomor[kunci] = label
        fig.savefig(FIG_DIR / f"fig_{bagian}_{nama}.png", bbox_inches="tight", dpi=150)
        plt.show()
        display(Markdown(f"**{label}.** {keterangan}"))
        return label

    def ref(self, kunci):
        """Nomor Gambar/Tabel yang sudah dibuat, untuk dikutip di narasi."""
        return self.nomor[kunci]
