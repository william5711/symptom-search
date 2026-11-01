"""Compare ranking methods on the hand-labelled queries in eval/queries.jsonl.

Usage:
    python evaluate.py                 # dev split, used for tuning
    python evaluate.py --split test    # held-out split, for final numbers
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from symptom_search import SymptomMatcher, load_diseases
from symptom_search.embeddings import DEFAULT_MODEL, Encoder, cosine_matrix

QUERIES_PATH = Path(__file__).resolve().parent / "eval" / "queries.jsonl"
Ranker = Callable[[str], List[str]]


def load_queries(split: str) -> List[dict]:
    with open(QUERIES_PATH, encoding="utf-8") as file:
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


def evaluate(rank: Ranker, queries: Sequence[dict]) -> Dict[str, float]:
    positions = []
    for query in queries:
        ranking = rank(query["query"])
        hits = [ranking.index(label) + 1 for label in query["labels"] if label in ranking]
        positions.append(min(hits) if hits else None)

    def hit_rate(k: int) -> float:
        return float(np.mean([p is not None and p <= k for p in positions]))

    return {
        "hit@1": hit_rate(1),
        "hit@3": hit_rate(3),
        "hit@10": hit_rate(10),
        "mrr": float(np.mean([1 / p if p else 0.0 for p in positions])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    diseases = load_diseases()
    queries = load_queries(args.split)
    encoder = Encoder(args.model)

    def matcher(mode: str) -> Ranker:
        return matcher_ranker(SymptomMatcher(diseases, encoder, mode=mode))

    rankers = {
        "TF-IDF keywords (baseline)": tfidf_ranker(diseases),
        "Original (pooler_output)": original_ranker(diseases, args.model),
        "Whole text (mean pooling)": matcher("whole_text"),
        "Clauses": matcher("clauses"),
        "Hybrid": matcher("hybrid"),
    }

    print(f"{len(queries)} queries, split={args.split}, model={args.model}\n")
    print(f"{'method':<30}{'hit@1':>8}{'hit@3':>8}{'hit@10':>8}{'MRR':>8}")
    for name, rank in rankers.items():
        scores = evaluate(rank, queries)
        print(f"{name:<30}" + "".join(f"{scores[k]:>8.3f}" for k in ("hit@1", "hit@3", "hit@10", "mrr")))


if __name__ == "__main__":
    main()
