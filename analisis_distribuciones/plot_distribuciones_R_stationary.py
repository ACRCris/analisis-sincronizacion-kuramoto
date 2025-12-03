"""Análisis y gráficas de distribuciones de R estacionario.

Genera:
- distribucion_R_total_TRAIN_MAC_60k.png
- distribucion_R_por_clase_TRAIN_MAC_60k.png

A partir de: resultados_kuramoto_TRAIN_MAC_60k/metricas_completas_TRAIN_MAC_60k.pt
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


RUTA_METRICAS = Path(
    "resultados_kuramoto_TRAIN_MAC_60k/metricas_completas_TRAIN_MAC_60k.pt"
)
RUTA_SALIDA = Path("resultados_kuramoto_TRAIN_MAC_60k/analisis_distribuciones")


def cargar_R_stationary_por_clase():
    """Carga R_stationary agrupado por clase desde el archivo de métricas."""

    datos = torch.load(RUTA_METRICAS, weights_only=False)["metricas"]

    R_por_clase = {i: [] for i in range(10)}
    for item in datos:
        label = int(item["label"])
        R_series = item["R_series"]
        R_final = float(R_series[-1].item() if hasattr(R_series[-1], "item") else R_series[-1])
        R_por_clase[label].append(R_final)

    for k in R_por_clase:
        R_por_clase[k] = np.array(R_por_clase[k], dtype=float)

    R_total = np.concatenate(list(R_por_clase.values()))
    return R_total, R_por_clase


def plot_R_total(R_total):
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sns.histplot(R_total, bins=50, kde=True, color="steelblue")
    plt.xlabel("R estacionario", fontsize=12)
    plt.ylabel("Frecuencia", fontsize=12)
    plt.title("Distribución de R estacionario (TRAIN MAC 60k)", fontsize=14, fontweight="bold")
    RUTA_SALIDA.mkdir(exist_ok=True, parents=True)
    out = RUTA_SALIDA / "distribucion_R_total_TRAIN_MAC_60k.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"✅ Guardado {out}")


def plot_R_por_clase(R_por_clase):
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharex=True, sharey=True)
    axes = axes.flatten()

    for clase in range(10):
        ax = axes[clase]
        datos = R_por_clase[clase]
        if len(datos) == 0:
            ax.set_visible(False)
            continue
        sns.histplot(datos, bins=40, kde=True, ax=ax, color=f"C{clase}")
        ax.set_title(f"Clase {clase} (N={len(datos)})", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle=":")

    fig.suptitle("Distribución de R estacionario por clase (TRAIN MAC 60k)", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    RUTA_SALIDA.mkdir(exist_ok=True, parents=True)
    out = RUTA_SALIDA / "distribucion_R_por_clase_TRAIN_MAC_60k.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"✅ Guardado {out}")


def main():
    print(f"📂 Cargando métricas desde: {RUTA_METRICAS}")
    R_total, R_por_clase = cargar_R_stationary_por_clase()
    print(f"  Total de muestras: {len(R_total)}")
    plot_R_total(R_total)
    plot_R_por_clase(R_por_clase)


if __name__ == "__main__":
    main()
