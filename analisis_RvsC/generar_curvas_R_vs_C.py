from pathlib import Path

from calcular_r_vs_c import main as _main


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    (base / "checkpoints").mkdir(parents=True, exist_ok=True)
    _main()
