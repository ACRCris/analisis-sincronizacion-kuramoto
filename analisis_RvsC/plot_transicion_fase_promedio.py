"""Transición de fase promedio R(c) en C in [0, 0.4].

Genera:
- transicion_fase_sincronizacion_0-0.4.png

Usa la base:
  codigo/analisis_criticalidad_minimalista/analisis_RvsC/R_vs_C.db
"""

from pathlib import Path
import argparse
import sqlite3
import pickle

import matplotlib.pyplot as plt
import numpy as np


DB_PATH = Path("R_vs_C/R_vs_C.db")
OUTPUT = Path("transicion_fase_sincronizacion_0-0.4.png")
C_MIN, C_MAX = 0.0, 0.4


def main():
    parser = argparse.ArgumentParser(
        description="Grafica la transición de fase promedio R(C) desde una base SQLite."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DB_PATH,
        help="Ruta al archivo SQLite con curvas R(C).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT,
        help="Ruta del PNG de salida.",
    )
    args = parser.parse_args()

    if not args.db_path.exists():
        parser.error(f"No existe la base de datos: {args.db_path}")

    conn = sqlite3.connect(str(args.db_path))
    cursor = conn.cursor()

    all_c = []
    all_r = []

    for clase in range(10):
        cursor.execute(f"SELECT c_values, r_values FROM clase_{clase}")
        rows = cursor.fetchall()
        for c_blob, r_blob in rows:
            c_vals = pickle.loads(c_blob)
            r_vals = pickle.loads(r_blob)
            mask = (c_vals >= C_MIN) & (c_vals <= C_MAX)
            c_f = c_vals[mask]
            r_f = r_vals[mask]
            if len(c_f) == 0:
                continue
            all_c.append(c_f)
            all_r.append(r_f)

    conn.close()

    c_grid = np.linspace(C_MIN, C_MAX, 300)
    r_interp = [np.interp(c_grid, c, r) for c, r in zip(all_c, all_r)]
    r_interp = np.vstack(r_interp)
    r_mean = r_interp.mean(axis=0)
    r_std = r_interp.std(axis=0)

    plt.figure(figsize=(10, 7))
    plt.plot(c_grid, r_mean, "b-", linewidth=2.5, label="R promedio")
    plt.fill_between(c_grid, r_mean - r_std, r_mean + r_std,
                     alpha=0.2, color="blue", label="±1 std")
    plt.xlabel("Acoplamiento (C)", fontsize=14, fontweight="bold")
    plt.ylabel("Pare1metro de orden final R(T)", fontsize=14, fontweight="bold")
    plt.title("Transicif3n de fase de sincronizacif3n (todas las clases)",
              fontsize=16, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.xlim(C_MIN, C_MAX)
    plt.ylim(0.0, 0.65)
    plt.legend(loc="lower right", fontsize=12)
    plt.tight_layout()
    args.output.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(args.output, dpi=300)
    plt.close()
    print(f"✅ Guardado {args.output}")


if __name__ == "__main__":
    main()
