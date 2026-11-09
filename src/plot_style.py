"""Gaya grafik sederhana untuk laporan (latar putih, garis tipis, grid samar)."""
import matplotlib.pyplot as plt

BLUE = "#2a78d6"     # seri pertama
ORANGE = "#eb6834"   # seri kedua (hanya jika membandingkan dua seri)
AQUA = "#1baf7a"     # seri ketiga
YELLOW = "#eda100"   # seri keempat
SERIES = [BLUE, ORANGE, AQUA, YELLOW]
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"


def setup():
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "legend.frameon": False,
    })
