"""
Run this in each venv before any pipeline run:
    python scripts/preflight.py --pipeline a
    python scripts/preflight.py --pipeline b

Exits with code 1 on any import failure so CI can catch it.
"""
import argparse
import sys


def check_pipeline_a() -> None:
    try:
        from transformers import ColPaliForRetrieval, ColPaliProcessor
    except ImportError as e:
        print(f"FAIL: ColPali import failed: {e}")
        print("Fix: bump transformers>=4.49.0 in envs/requirements_a.txt and reinstall.")
        sys.exit(1)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        print(f"FAIL: sentence-transformers import failed: {e}")
        sys.exit(1)

    try:
        import torch
        import qwen_vl_utils  # noqa: F401
    except ImportError as e:
        print(f"FAIL: {e}")
        sys.exit(1)

    print("Pipeline A imports OK")
    print(f"  transformers: {__import__('transformers').__version__}")
    print(f"  torch: {torch.__version__}")


def check_pipeline_b() -> None:
    try:
        from transformers import AutoModel, AutoTokenizer
        import torch
    except ImportError as e:
        print(f"FAIL: {e}")
        sys.exit(1)

    try:
        import transformers
        ver = transformers.__version__
        major, minor, *_ = ver.split(".")
        if not (int(major) == 4 and int(minor) == 40):
            print(f"WARN: transformers=={ver}; expected 4.40.x. "
                  "VisRAG-Ret trust_remote_code is sensitive to this pin.")
    except Exception:
        pass

    print("Pipeline B imports OK")
    print(f"  transformers: {__import__('transformers').__version__}")
    print(f"  torch: {torch.__version__}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=["a", "b"], required=True)
    args = parser.parse_args()

    if args.pipeline == "a":
        check_pipeline_a()
    else:
        check_pipeline_b()
