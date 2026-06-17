"""Check the local project environment.

Run:
    python scripts/check_environment.py
"""

from pathlib import Path
import importlib.util
import platform
import sys


REQUIRED_PACKAGES = [
    "numpy",
    "pandas",
    "matplotlib",
    "sklearn",
    "datasets",
    "transformers",
    "torch",
    "yaml",
    "tqdm",
    "joblib",
]


def package_available(package_name: str) -> bool:
    """Return True if a package can be imported."""
    return importlib.util.find_spec(package_name) is not None


def main() -> None:
    """Print environment information."""
    project_root = Path(__file__).resolve().parents[1]

    print("Project root:", project_root)
    print("Python:", sys.version)
    print("Platform:", platform.platform())

    print("\nPackage check:")
    for package in REQUIRED_PACKAGES:
        status = "OK" if package_available(package) else "MISSING"
        print(f"  {package:<15} {status}")

    print("\nTorch device check:")
    if package_available("torch"):
        import torch

        print("  torch version:", torch.__version__)
        print("  CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("  CUDA device:", torch.cuda.get_device_name(0))
        else:
            print("  Device: CPU for local development")
    else:
        print("  torch is not installed")

    print("\nFolder check:")
    expected_dirs = [
        "app/backend",
        "app/frontend",
        "configs",
        "data/raw",
        "data/processed",
        "data/noisy",
        "data/audit",
        "notebooks",
        "reports/figures",
        "reports/tables",
        "scripts",
        "src",
    ]

    for relative_dir in expected_dirs:
        path = project_root / relative_dir
        status = "OK" if path.exists() else "MISSING"
        print(f"  {relative_dir:<30} {status}")


if __name__ == "__main__":
    main()
