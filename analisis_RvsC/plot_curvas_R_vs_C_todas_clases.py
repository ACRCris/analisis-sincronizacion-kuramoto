"""Curvas R vs C para todas las clases (0-9) en [0, 0.4].

Genera:
- curvas_todas_clases_excepto_0-0.4.png

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
OUTPUT = Path("curvas_todas_clases_excepto_0-0.4.png")


def main():
    parser = argparse.ArgumentParser(
        description="Grafica todas las curvas R vs C a partir de una base SQLite."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DB_PATH,
        help="Ruta al archivo SQLite con tablas clase_0..clase_9.",
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

    plt.figure(figsize=(12, 8))
    colores = plt.cm.tab10(np.linspace(0, 1, 10))
    total_curvas = 0

    for clase in range(10):
        cursor.execute(f"SELECT c_values, r_values FROM clase_{clase}")
        rows = cursor.fetchall()
        for c_blob, r_blob in rows:
            c_vals = pickle.loads(c_blob)
            r_vals = pickle.loads(r_blob)
            mask = (c_vals >= 0.0) & (c_vals <= 0.4)
            c_f = c_vals[mask]
            r_f = r_vals[mask]
            if len(c_f) == 0:
                continue
            plt.plot(c_f, r_f, alpha=0.08, color=colores[clase])
            total_curvas += 1

    conn.close()

    plt.title("Todas las curvas R vs C (clases 0-9), C in [0, 0.4]",
              fontsize=14, fontweight="bold")
    plt.xlabel("C (acoplamiento)", fontsize=12)
    plt.ylabel("R (pare1metro de orden)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(0.0, 0.4)
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    args.output.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(args.output, dpi=300)
    plt.close()
    print(f"✅ Guardado {args.output} (total_curvas={total_curvas})")


if __name__ == "__main__":
    main()
