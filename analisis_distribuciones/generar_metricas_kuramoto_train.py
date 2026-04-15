from pathlib import Path

from run_kuramoto_TRAIN_MAC import main as _main


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    local_results = base / "resultados_kuramoto_TRAIN_MAC_60k"
    local_results.mkdir(parents=True, exist_ok=True)
    _main()
