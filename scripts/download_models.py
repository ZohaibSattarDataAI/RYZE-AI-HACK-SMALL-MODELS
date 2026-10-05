"""
One-time model download for FinGuard AI.
Run once with internet. After that, the app is fully offline.
"""

import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPLAINER_DIR = os.path.join(ROOT, "models", "explainer")

QWEN_URL = (
    "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/"
    "qwen2.5-0.5b-instruct-q4_k_m.gguf"
)
QWEN_FILE = "qwen2.5-0.5b-instruct-q4_k_m.gguf"


def _download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.isfile(dest):
        print(f"[skip] {dest} already exists")
        return
    print(f"[downloading] {url}")
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while True:
            buf = r.read(1024 * 64)
            if not buf:
                break
            f.write(buf)
            done += len(buf)
            if total:
                pct = done * 100 / total
                sys.stdout.write(f"\r  {pct:5.1f}%  {done/1e6:.1f} MB")
                sys.stdout.flush()
    print("\n  done")


if __name__ == "__main__":
    _download(QWEN_URL, os.path.join(EXPLAINER_DIR, QWEN_FILE))