from pathlib import Path

from calcular_R_critico_local import main as _main


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    (base / "resultados_c_critical").mkdir(parents=True, exist_ok=True)
    _main()
