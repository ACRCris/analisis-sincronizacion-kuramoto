#!/usr/bin/env python3
"""
Prepara MNIST para Analisis_Sincronizacion.

Replica el flujo de setup_mnist.py usado en codigo:
1) Crea la ruta local de datos
2) Llama datasets.MNIST(download=True) para train y test
3) Verifica integridad con un batch de prueba

Adicionalmente:
- Si la descarga remota falla (por ejemplo, SSL), intenta usar un origen local
  de archivos raw y copiarlo a la ruta objetivo.
- Sincroniza rutas de compatibilidad usadas por scripts legacy.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
from torchvision import datasets, transforms

RAW_FILENAMES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
]


def has_complete_mnist_raw(raw_dir: Path) -> bool:
    return all((raw_dir / name).exists() for name in RAW_FILENAMES)


def copy_raw_files(src_raw: Path, dst_raw: Path) -> None:
    dst_raw.mkdir(parents=True, exist_ok=True)
    for name in RAW_FILENAMES:
        shutil.copy2(src_raw / name, dst_raw / name)


def load_mnist(root: Path, download: bool):
    tf = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(str(root), train=True, download=download, transform=tf)
    test = datasets.MNIST(str(root), train=False, download=download, transform=tf)
    return train, test


def verify_batch(trainset) -> None:
    loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False)
    x, y = next(iter(loader))
    print("      ✓ Batch de prueba cargado correctamente")
    print(f"      - Shape entrada: {tuple(x.shape)}")
    print(f"      - Shape etiquetas: {tuple(y.shape)}")
    print(f"      - Rango valores: [{float(x.min()):.3f}, {float(x.max()):.3f}]")


def find_local_source(base_dir: Path, primary_raw: Path, local_raw_source: Path | None = None):
    candidates = []
    if local_raw_source is not None:
        candidates.append(local_raw_source)

    candidates.extend(
        [
            base_dir / "data" / "MNIST" / "raw",
            base_dir.parent / "data" / "MNIST" / "raw",
            base_dir.parent / "codigo" / "data" / "MNIST" / "raw",
            base_dir.parent.parent / "data" / "MNIST" / "raw",
        ]
    )

    seen = set()
    for c in candidates:
        try:
            resolved = c.resolve()
        except FileNotFoundError:
            resolved = c
        if resolved in seen:
            continue
        seen.add(resolved)
        if c == primary_raw:
            continue
        if c.exists() and has_complete_mnist_raw(c):
            return c
    return None


def sync_compat_roots(primary_root: Path, base_dir: Path, compat_roots: list[Path] | None = None) -> None:
    primary_raw = primary_root / "MNIST" / "raw"
    if compat_roots is None:
        compat_roots = [
            base_dir.parent / "data",
            base_dir.parent.parent / "data",
        ]

    print("\nSincronizando rutas de compatibilidad...")
    for root in compat_roots:
        dst_raw = root / "MNIST" / "raw"
        try:
            if root.resolve() == primary_root.resolve():
                print(f"      - Saltando (misma ruta): {root}")
                continue
        except FileNotFoundError:
            pass

        try:
            copy_raw_files(primary_raw, dst_raw)
            load_mnist(root, download=False)
            print(f"      ✓ Listo: {root}")
        except Exception as e:
            print(f"      ⚠ No se pudo sincronizar {root}: {e}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Configura MNIST para Analisis_Sincronizacion")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Directorio raíz donde se preparará MNIST (default: Analisis_Sincronizacion/data).",
    )
    parser.add_argument(
        "--local-raw-source",
        type=Path,
        default=None,
        help="Ruta opcional a un directorio MNIST/raw local para fallback.",
    )
    parser.add_argument(
        "--compat-roots",
        type=Path,
        nargs="*",
        default=None,
        help="Rutas opcionales de sincronización de compatibilidad.",
    )
    parser.add_argument(
        "--no-sync-compat",
        action="store_true",
        help="No sincronizar rutas de compatibilidad fuera de Analisis_Sincronizacion/data",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    primary_root = args.data_root if args.data_root is not None else base_dir / "data"
    primary_raw = primary_root / "MNIST" / "raw"

    print("=" * 80)
    print("CONFIGURANDO MNIST PARA ANALISIS_SINCRONIZACION")
    print("=" * 80)
    print(f"\n[1/3] Preparando ruta principal: {primary_root}")
    primary_raw.mkdir(parents=True, exist_ok=True)

    trainset = testset = None

    print("\n[2/3] Descargando/cargando MNIST con torchvision...")
    if has_complete_mnist_raw(primary_raw):
        print("      ✓ Archivos raw ya presentes en la ruta principal")
        trainset, testset = load_mnist(primary_root, download=False)
    else:
        try:
            trainset, testset = load_mnist(primary_root, download=True)
            print("      ✓ Descarga/carga remota completada")
        except Exception as e:
            print(f"      ⚠ Descarga remota fallo: {e}")
            print("      → Intentando origen local de respaldo...")
            src_raw = find_local_source(base_dir, primary_raw, args.local_raw_source)
            if src_raw is None:
                print("      ✗ No se encontro origen local completo de MNIST/raw")
                return 1
            print(f"      ✓ Usando origen local: {src_raw}")
            copy_raw_files(src_raw, primary_raw)
            trainset, testset = load_mnist(primary_root, download=False)
            print("      ✓ Carga local completada")

    print("\n[3/3] Verificando integridad...")
    print(f"      ✓ Entrenamiento: {len(trainset)} imagenes")
    print(f"      ✓ Validacion: {len(testset)} imagenes")
    verify_batch(trainset)

    if not args.no_sync_compat:
        sync_compat_roots(primary_root, base_dir, args.compat_roots)

    print("\n" + "=" * 80)
    print("MNIST CONFIGURADO EXITOSAMENTE")
    print("=" * 80)
    print(f"\nRuta principal: {primary_root / 'MNIST' / 'raw'}")
    if args.no_sync_compat:
        print("Rutas de compatibilidad: desactivadas por bandera --no-sync-compat")
    else:
        print("Rutas de compatibilidad: activadas")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
