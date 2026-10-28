"""Shared configuration and file helpers; unavailable to the isolated learner."""
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SPLITS = ("train", "validation", "test_iid", "test_ood", "test_modes")
OUTPUT_FILES = ("result.json", "dataset.npz", "model.pt", "training.csv", "loss.png", "report.md")


def load_config():
    path = PROJECT_ROOT/"config.json"
    if not path.is_file():
        path = Path(sys.prefix)/"share/neural-relations/config.json"
    return json.loads(path.read_text())


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
