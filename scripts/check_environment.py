"""Read-only environment checks. Never install or download anything."""
import argparse
import importlib
from pathlib import Path
import struct
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deep", action="store_true", help="Require a TensorFlow CPU computation too.")
    parser.add_argument("--strict-pins", action="store_true", help="Fail if the selected direct dependency pins differ.")
    args = parser.parse_args()
    print("Python:", sys.version)
    print("Interpreter:", sys.executable)
    print("Architecture:", struct.calcsize("P") * 8, "bit")
    failures = []
    if sys.version_info[:2] not in {(3, 12), (3, 13)} or struct.calcsize("P") * 8 != 64:
        failures.append("Use standard 64-bit Python 3.12 (or supported 3.13).")
    for name in ["numpy", "scipy", "pandas", "sklearn", "seaborn", "matplotlib", "ml_lib", "lab_helpers", "nbformat", "nbclient", "ipykernel", "jupyterlab", "ai_lib", "langchain_core", "langgraph", "langchain_google_genai", "langchain_groq", "langchain_cohere", "langchain_tavily", "rank_bm25", "fastembed", "qdrant_client", "dotenv", "langchain", "langsmith", "openevals"]:
        try:
            module = importlib.import_module(name)
            print("OK", name, getattr(module, "__version__", "importable"))
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    if args.deep:
        try:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            import tensorflow as tf
            tf.config.set_visible_devices([], "GPU")
            value = float(tf.reduce_sum(tf.constant([1.0, 2.0])).numpy())
            if value != 3.0: raise RuntimeError("Unexpected TensorFlow arithmetic result.")
            print("OK TensorFlow", tf.__version__, "CPU computation:", value)
        except Exception as exc:
            failures.append(f"TensorFlow: {type(exc).__name__}: {exc}")
    from importlib import metadata
    requirements = Path(__file__).resolve().parents[1] / "requirements.txt"
    for line in requirements.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "==" not in line: continue
        name, wanted = line.split("==", 1)
        if not args.deep and name in {"tensorflow", "keras"}: continue
        try: actual = metadata.version(name)
        except metadata.PackageNotFoundError: actual = "not installed"
        if actual != wanted:
            message = f"PIN differs: {name}: expected {wanted}, found {actual}"
            print(message)
            if args.strict_pins: failures.append(message)
    if failures:
        print("\nChecks needing attention:")
        for failure in failures: print(" -", failure)
        return 1
    print("\nRequested checks passed. This is not a model-training or real-data quality validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
