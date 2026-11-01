"""Compare embedding models in the hybrid matcher.

Similarity scales differ between models, so each one gets its own thresholds, tuned on
the dev split and saved to eval/model_settings.json. The test split only reads them.

    python scripts/compare_models.py                # tune on dev, save settings
    python scripts/compare_models.py --split test   # held-out numbers, settings unchanged
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate import (
    answer_position,
    bootstrap_interval,
    hit_rate,
    holm_adjust,
    load_queries,
    matcher_ranker,
    reciprocal_ranks,
    sign_flip_p_value,
)
from symptom_search import SymptomMatcher, load_diseases
from symptom_search.embeddings import Encoder, cosine_matrix
from symptom_search.text import is_negated, split_clauses

SETTINGS_PATH = Path(__file__).resolve().parent.parent / "eval" / "model_settings.json"
BASELINE = "sentence-transformers/all-MiniLM-L6-v2"
MODELS = [
    BASELINE,
    "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "intfloat/e5-small-v2",
    "intfloat/e5-base-v2",
    "dmis-lab/biobert-base-cased-v1.2",
    "NeuML/pubmedbert-base-embeddings",
    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
]
SIMILARITY_GRID = [round(x, 2) for x in np.arange(0.20, 0.86, 0.05)]
QUERY_SETS = ("written", "ddxplus")


class MemoEncoder:
    """Remembers query vectors, since tuning repeats the same queries many times."""

    def __init__(self, encoder: Encoder):
        self.encoder = encoder
        self.encode_cached = encoder.encode_cached
        self._memo: Dict[tuple, np.ndarray] = {}

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        key = tuple(texts)
        if key not in self._memo:
            self._memo[key] = self.encoder.encode(texts)
        return self._memo[key]


def positions_for(matcher: SymptomMatcher, queries: Sequence[dict]) -> List[Optional[int]]:
    rank = matcher_ranker(matcher)
    return [answer_position(rank(q["query"]), q["labels"]) for q in queries]


def best_similarity(matcher: SymptomMatcher, text: str) -> float:
    clauses = [clause for clause in split_clauses(text) if not is_negated(clause)]
    if not clauses:
        return 0.0
    return float(cosine_matrix(matcher.encoder.encode(clauses), matcher._symptom_vectors).max())


def tune(diseases, encoder: MemoEncoder, queries: Dict[str, List[dict]], unrelated: List[dict]) -> dict:
    """Pick min_similarity by mean dev MRR, then the lowest floor that rejects all unrelated dev text."""
    def mean_mrr(min_similarity: float) -> float:
        matcher = SymptomMatcher(diseases, encoder, min_similarity=min_similarity, min_query_similarity=0.0)
        return float(np.mean([reciprocal_ranks(positions_for(matcher, queries[s])).mean() for s in QUERY_SETS]))

    min_similarity = max(SIMILARITY_GRID, key=mean_mrr)
    matcher = SymptomMatcher(diseases, encoder, min_similarity=min_similarity, min_query_similarity=0.0)
    highest = max(best_similarity(matcher, q["query"]) for q in unrelated)
    return {"min_similarity": min_similarity, "min_query_similarity": math.ceil(highest * 20) / 20}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    args = parser.parse_args()

    diseases = load_diseases()
    queries = {name: load_queries(args.split, name) for name in QUERY_SETS}
    unrelated = load_queries(args.split, "unrelated")
    settings = json.loads(SETTINGS_PATH.read_text()) if args.split == "test" else {}

    baseline_ranks: Dict[str, np.ndarray] = {}
    pooled_gains: Dict[str, np.ndarray] = {}
    print(f"split={args.split}; each cell is hit@3 / hit@10 / MRR, then MRR gain over {BASELINE.split('/')[1]} with a 95% interval\n", flush=True)
    for model in MODELS:
        encoder = MemoEncoder(Encoder(model))
        if args.split == "dev":
            settings[model] = tune(diseases, encoder, queries, unrelated)
            SETTINGS_PATH.write_text(json.dumps(settings, indent=2) + "\n")

        chosen = settings[model]
        matcher = SymptomMatcher(diseases, encoder, **chosen)
        no_floor = SymptomMatcher(diseases, encoder, min_similarity=chosen["min_similarity"], min_query_similarity=0.0)
        answered = sum(bool(matcher.search(q["query"])) for q in unrelated)
        print(f"{model}  {chosen}  unrelated answered {answered}/{len(unrelated)}")
        gains = []
        for name in QUERY_SETS:
            positions = positions_for(matcher, queries[name])
            ranks = reciprocal_ranks(positions)
            refused = sum(not matcher.search(q["query"]) for q in queries[name])
            baseline_ranks.setdefault(name, ranks)
            gains.append(ranks - baseline_ranks[name])
            low, high = bootstrap_interval(ranks - baseline_ranks[name])
            no_floor_mrr = reciprocal_ranks(positions_for(no_floor, queries[name])).mean()
            print(
                f"  {name:<8} {hit_rate(positions, 3):.3f} / {hit_rate(positions, 10):.3f} / {ranks.mean():.3f}"
                f"  gain {(ranks - baseline_ranks[name]).mean():+.3f} [{low:+.3f}, {high:+.3f}]"
                f"  refused {refused}/{len(positions)}  MRR without floor {no_floor_mrr:.3f}",
                flush=True,
            )
        if model != BASELINE:
            pooled_gains[model] = np.concatenate(gains)

    # Each query counts once, across both sets.
    p_values = [sign_flip_p_value(gain) for gain in pooled_gains.values()]
    print(f"\nMRR gain over {BASELINE.split('/')[1]} on both sets together, Holm-corrected for {len(p_values)} models:")
    for (model, gain), raw, adjusted in zip(pooled_gains.items(), p_values, holm_adjust(p_values)):
        low, high = bootstrap_interval(gain)
        print(f"  {model:<48} {gain.mean():+.3f} [{low:+.3f}, {high:+.3f}]  p={raw:.4f}  corrected p={adjusted:.4f}")


if __name__ == "__main__":
    main()
