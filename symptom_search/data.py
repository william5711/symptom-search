"""Load the symptom-disease table built by scripts/build_dataset.py."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass(frozen=True)
class Disease:
    name: str
    symptoms: Tuple[str, ...]


def load_diseases(path: Union[str, Path] = DATA_DIR / "diseases.csv") -> List[Disease]:
    symptoms: Dict[str, List[str]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            symptoms[row["disease"]].append(row["symptom"])
    return [Disease(name, tuple(listed)) for name, listed in sorted(symptoms.items())]
