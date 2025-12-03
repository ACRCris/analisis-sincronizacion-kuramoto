"""Gráficas de distribuciones de c_critico y R_critico.

Genera las figuras finales usadas en el documento:
- distribucion_c_critico_por_clase.png
- distribucion_c_critico_resumen copia 2.png
- distribucion_R_critico_total.png

A partir de la base SQLite en
  codigo/analisis_criticalidad_minimalista/analisis_alpha_c/resultados_c_critical/mnist_c_critical.db
  y
  codigo/analisis_criticalidad_minimalista/analisis_alpha_c/resultados_c_critical/mnist_R_critico.db
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sqlite3


BASE_DIR = Path("resultados_c_critical")
DB_C = BASE_DIR / "mnist_c_critical.db"
DB_R = BASE_DIR / "mnist_R_critico.db"


def cargar_c_critico_por_clase():
    conn = sqlite3.connect(str(DB_C))
    cursor = conn.cursor()
    c_por_clase = {}
    for clase in range(10):
        cursor.execute(f"SELECT c_critico FROM clase_{clase}")
        vals = [row[0] for row in cursor.fetchall()]
        c_por_clase[clase] = np.array(vals, dtype=float)
    conn.close()
    return c_por_clase


def cargar_R_critico():
    conn = sqlite3.connect(str(DB_R))
    cursor = conn.cursor()
    R_global = []
    for clase in range(10):
        cursor.execute(f"SELECT R_critico FROM clase_{clase}")
        R_global.extend(row[0] for row in cursor.fetchall())
    conn.close()
    return np.array(R_global, dtype=float)


def plot_c_critico_por_clase(c_por_clase):
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharex=True, sharey=True)
    axes = axes.flatten()

    for clase in range(10):
        ax = axes[clase]
        datos = c_por_clase[clase]
        if len(datos) == 0:
            ax.set_visible(False)
            continue
        sns.histplot(datos, bins=40, kde=True, ax=ax, color=f"C{clase}")
        ax.set_title(f"Clase {clase} (N={len(datos)})", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle=":")

    fig.suptitle("Distribución de c_critico por clase", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = BASE_DIR / "distribucion_c_critico_por_clase.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"✅ Guardado {out}")


def plot_c_critico_resumen(c_por_clase):
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    medios = [np.mean(c_por_clase[k]) for k in range(10)]
    stds = [np.std(c_por_clase[k]) for k in range(10)]
    clases = np.arange(10)
    plt.errorbar(clases, medios, yerr=stds, fmt="o", capsize=4)
    plt.xticks(clases)
    plt.xlabel("Clase", fontsize=12)
    plt.ylabel("c_critico", fontsize=12)
    plt.title("Resumen de c_critico por clase", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3, linestyle=":")
    out = BASE_DIR / "distribucion_c_critico_resumen copia 2.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"✅ Guardado {out}")


def plot_R_critico_total(R_total):
    from matplotlib.ticker import FuncFormatter

    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sns.histplot(R_total, bins=50, kde=True, color="steelblue")
    plt.xlabel("R_critico", fontsize=12)
    plt.ylabel("Frecuencia", fontsize=12)
    plt.title("Distribución total de R_critico", fontsize=14, fontweight="bold")

    def fmt(x, pos):
        return f"{x:.2f}"

    plt.gca().xaxis.set_major_formatter(FuncFormatter(fmt))
    out = BASE_DIR / "distribucion_R_critico_total.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"✅ Guardado {out}")


def main():
    print(f"📂 Usando DB c_critico: {DB_C}")
    print(f"📂 Usando DB R_critico: {DB_R}")
    c_por_clase = cargar_c_critico_por_clase()
    R_total = cargar_R_critico()
    print(f"  Total de muestras R_critico: {len(R_total)}")
    plot_c_critico_por_clase(c_por_clase)
    plot_c_critico_resumen(c_por_clase)
    plot_R_critico_total(R_total)


if __name__ == "__main__":
    main()
