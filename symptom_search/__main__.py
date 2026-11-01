"""Command line interface: python -m symptom_search "I have a fever and a sore throat"."""

from __future__ import annotations

import argparse

from . import SymptomMatcher, load_diseases
from .embeddings import Encoder

DISCLAIMER = "Educational project, not medical advice. See a clinician about real symptoms."


def main() -> None:
    parser = argparse.ArgumentParser(description="Find diseases that match a description of symptoms.")
    parser.add_argument("text", nargs="?", help="symptom description (prompts for it if omitted)")
    parser.add_argument("-k", "--top-k", type=int, default=5, help="number of results to show")
    args = parser.parse_args()

    text = args.text or input("Describe your symptoms: ")
    matcher = SymptomMatcher(load_diseases(), Encoder())
    matches = matcher.search(text, top_k=args.top_k)

    print(f"\n{DISCLAIMER}\n")
    if not matches:
        print("No known symptoms recognised. Try describing what you feel in more detail.")
        return

    for position, match in enumerate(matches, 1):
        print(f"{position}. {match.disease}")
        for item in match.evidence:
            print(f'     "{item.clause}" -> {item.symptom} ({item.similarity:.2f})')


if __name__ == "__main__":
    main()
