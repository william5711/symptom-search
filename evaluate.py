"""Compare ranking methods on the labelled queries in eval/.

Usage:
    python evaluate.py                       # written queries, dev split
    python evaluate.py --queries ddxplus     # synthetic patients from DDXPlus
    python evaluate.py --errors              # dev split: list what the hybrid missed
    python evaluate.py --split test          # held-out split, for final numbers
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from symptom_search import SymptomMatcher, load_diseases
from symptom_search.embeddings import DEFAULT_MODEL, Encoder, cosine_matrix

EVAL_DIR = Path(__file__).resolve().parent / "eval"
QUERY_SETS = {
    "written": EVAL_DIR / "queries.jsonl",
    "ddxplus": EVAL_DIR / "ddxplus_queries.jsonl",
    "unrelated": EVAL_DIR / "unrelated.jsonl",
}
Ranker = Callable[[str], List[str]]
Interval = Tuple[float, float]


def load_queries(split: str, query_set: str = "written") -> List[dict]:
    with open(QUERY_SETS[query_set], encoding="utf-8") as file:
        queries = [json.loads(line) for line in file if line.strip()]
    return [q for q in queries if split == "all" or q["split"] == split]


def rank_by_similarity(query_vector: np.ndarray, matrix: np.ndarray, names: Sequence[str]) -> List[str]:
    return [names[i] for i in np.argsort(-cosine_matrix(query_vector[None, :], matrix)[0])]


def tfidf_ranker(diseases) -> Ranker:
    """Keyword baseline: no embeddings at all."""
    names = [d.name for d in diseases]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform(" ".join(d.symptoms) for d in diseases)
    return lambda q: [names[i] for i in np.argsort(-(matrix @ vectorizer.transform([q]).T).toarray().ravel())]


def original_ranker(diseases, model_name: str = DEFAULT_MODEL) -> Ranker:
    """The method from legacy/final.py: pooler_output of `" and ".join(symptoms)`."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    def embed(texts: Sequence[str]) -> np.ndarray:
        encoded = tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            return F.normalize(model(**encoded).pooler_output, p=2, dim=1).numpy()

    names = [d.name for d in diseases]
    matrix = np.vstack([embed([" and ".join(d.symptoms)]) for d in diseases])
    return lambda q: rank_by_similarity(embed([q])[0], matrix, names)


def matcher_ranker(matcher: SymptomMatcher) -> Ranker:
    return lambda q: [m.disease for m in matcher.search(q, top_k=len(matcher.diseases))]


def answer_position(ranking: Sequence[str], labels: Sequence[str]) -> Optional[int]:
    """1-based rank of the best-placed correct entry, or None if there isn't one."""
    hits = [ranking.index(label) + 1 for label in labels if label in ranking]
    return min(hits) if hits else None


def hit_rate(positions: Sequence[Optional[int]], k: int) -> float:
    return float(np.mean([p is not None and p <= k for p in positions]))


def reciprocal_ranks(positions: Sequence[Optional[int]]) -> np.ndarray:
    return np.array([1 / p if p else 0.0 for p in positions])


def wilson_interval(hits: int, n: int, z: float = 1.96) -> Interval:
    """95% interval for a hit rate. Unlike the normal approximation, it stays inside 0-1."""
    rate = hits / n
    denominator = 1 + z ** 2 / n
    center = (rate + z ** 2 / (2 * n)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / n + z ** 2 / (4 * n ** 2)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def bootstrap_interval(values: np.ndarray, resamples: int = 5000, seed: int = 0) -> Interval:
    """95% percentile interval for a mean, resampling queries."""
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(resamples, len(values)), replace=True).mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def sign_flip_p_value(differences: np.ndarray, resamples: int = 20000, seed: int = 0) -> float:
    """Two-sided paired test: how often random sign flips give a mean this far from zero."""
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1, 1], size=(resamples, len(differences)))
    flipped = np.abs((signs * differences).mean(axis=1))
    return float((1 + np.sum(flipped >= abs(differences.mean()) - 1e-12)) / (resamples + 1))


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    """Holm correction, so testing many models doesn't make a lucky winner look real."""
    order = np.argsort(p_values)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for position, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - position) * p_values[index]))
        adjusted[index] = running
    return adjusted


def print_results(all_positions: Dict[str, List[Optional[int]]], baseline: str) -> None:
    print(f"{'method':<28}{'hit@1':>8}{'hit@3':>8}{'hit@3 95% CI':>16}{'hit@10':>8}{'MRR':>8}{'MRR 95% CI':>16}{'MRR vs baseline':>22}")
    base = reciprocal_ranks(all_positions[baseline])
    for name, positions in all_positions.items():
        n = len(positions)
        low3, high3 = wilson_interval(round(hit_rate(positions, 3) * n), n)
        ranks = reciprocal_ranks(positions)
        low_mrr, high_mrr = bootstrap_interval(ranks)
        gain = ""
        if name != baseline:
            low, high = bootstrap_interval(ranks - base)
            gain = f"{(ranks - base).mean():+.3f} [{low:+.3f}, {high:+.3f}]"
        print(
            f"{name:<28}{hit_rate(positions, 1):>8.3f}{hit_rate(positions, 3):>8.3f}"
            f"{f'[{low3:.2f}, {high3:.2f}]':>16}{hit_rate(positions, 10):>8.3f}{ranks.mean():>8.3f}"
            f"{f'[{low_mrr:.2f}, {high_mrr:.2f}]':>16}{gain:>22}"
        )


def print_errors(rank: Ranker, queries: Sequence[dict]) -> None:
    """List queries whose answer isn't in the top 3, worst group first."""
    no_entry, outside_top_10, outside_top_3 = [], [], []
    for query in queries:
        ranking = rank(query["query"])
        position = answer_position(ranking, query["labels"])
        row = (position, query, ranking[:3])
        if not query["labels"]:
            no_entry.append(row)
        elif position is None or position > 10:
            outside_top_10.append(row)
        elif position > 3:
            outside_top_3.append(row)

    groups = [("No entry in the dataset", no_entry), ("Outside the top 10", outside_top_10), ("Ranked 4 to 10", outside_top_3)]
    for title, rows in groups:
        print(f"\n{title}: {len(rows)} of {len(queries)}")
        for position, query, top in rows:
            wanted = query.get("condition") or ", ".join(query["labels"]) or "(unlabelled)"
            print(f"  [{position or '-'}] {wanted}: {query['query'][:110]}")
            print(f"      got: {' | '.join(top)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    parser.add_argument("--queries", default="written", choices=["written", "ddxplus"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--errors", action="store_true", help="list misses by the hybrid (dev split only)")
    args = parser.parse_args()
    if args.errors and args.split != "dev":
        parser.error("--errors only runs on dev, so test queries stay unseen")

    diseases = load_diseases()
    queries = load_queries(args.split, args.queries)
    encoder = Encoder(args.model)

    def matcher(mode: str) -> Ranker:
        return matcher_ranker(SymptomMatcher(diseases, encoder, mode=mode))

    baseline = "TF-IDF keywords (baseline)"
    rankers = {
        baseline: tfidf_ranker(diseases),
        "Original (pooler_output)": original_ranker(diseases, args.model),
        "Whole text (mean pooling)": matcher("whole_text"),
        "Clauses": matcher("clauses"),
        "Hybrid": matcher("hybrid"),
    }

    no_entry = sum(not q["labels"] for q in queries)
    print(f"{len(queries)} queries ({no_entry} with no entry), set={args.queries}, split={args.split}, model={args.model}\n")
    print_results(
        {name: [answer_position(rank(q["query"]), q["labels"]) for q in queries] for name, rank in rankers.items()},
        baseline,
    )
    unrelated = load_queries(args.split, "unrelated")
    answered = sum(bool(rankers["Hybrid"](q["query"])) for q in unrelated)
    print(f"\nUnrelated text the hybrid still answered: {answered} of {len(unrelated)}")
    if args.errors:
        print_errors(rankers["Hybrid"], queries)


if __name__ == "__main__":
    main()
