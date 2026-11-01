"""Fine-tune an embedding model so symptoms of the same condition sit closer together.

Each batch holds one pair of symptoms from each of several conditions. A symptom should
be closest to its partner, and the other pairs in the batch act as negatives. Pairs that
cross sources come first, because they teach that plain wording ("Feeling very thirsty")
and clinical wording ("polydipsia") can mean the same thing.

Entries labelled in a test split are left out of training. Training shifts the similarity
scale, so every epoch is scored with min_similarity re-tuned on dev, and the best epoch is
chosen on dev. The model is saved under .cache/ and isn't part of the repo.

Usage: python scripts/finetune.py --base BAAI/bge-base-en-v1.5 --run all-pairs
"""

from __future__ import annotations

import argparse
import csv
import itertools
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate import answer_position, load_queries, matcher_ranker, reciprocal_ranks
from symptom_search import SymptomMatcher, load_diseases
from symptom_search.embeddings import DEFAULT_CACHE_DIR, MODEL_PREFIXES

ROOT = Path(__file__).resolve().parent.parent
SEED = 20261101
MAX_PAIRS_PER_ENTRY = 20
TEMPERATURE = 0.05
QUERY_SETS = ("written", "ddxplus")
Pair = Tuple[str, str]


def held_out_entries() -> Set[str]:
    return {label for name in QUERY_SETS for query in load_queries("test", name) for label in query["labels"]}


def training_pairs(
    rows: Sequence[dict], excluded: Set[str], rng: random.Random, cross_source_only: bool = False
) -> Dict[str, List[Pair]]:
    """Map each training entry to its symptom pairs, cross-source pairs first."""
    by_entry: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        if row["disease"] not in excluded:
            by_entry[row["disease"]].append(row)

    pairs = {}
    for entry, symptoms in by_entry.items():
        combos = list(itertools.combinations(symptoms, 2))
        rng.shuffle(combos)
        combos.sort(key=lambda combo: combo[0]["sources"] == combo[1]["sources"])
        if cross_source_only:
            combos = [combo for combo in combos if combo[0]["sources"] != combo[1]["sources"]]
        if combos:
            pairs[entry] = [(a["symptom"], b["symptom"]) for a, b in combos[:MAX_PAIRS_PER_ENTRY]]
    return pairs


def batches(pairs: Dict[str, List[Pair]], batch_size: int, rng: random.Random) -> List[List[Pair]]:
    """One epoch of batches, with at most one pair from any entry in a batch."""
    remaining = {entry: rng.sample(entry_pairs, len(entry_pairs)) for entry, entry_pairs in pairs.items()}
    result = []
    while len(remaining) >= batch_size:
        chosen = rng.sample(sorted(remaining), batch_size)
        result.append([remaining[entry].pop() for entry in chosen])
        remaining = {entry: left for entry, left in remaining.items() if left}
    return result


def contrastive_loss(left, right, temperature: float = TEMPERATURE):
    """Cross-entropy that makes each left vector pick out its own right vector."""
    import torch
    import torch.nn.functional as F

    logits = F.normalize(left, dim=1) @ F.normalize(right, dim=1).T / temperature
    targets = torch.arange(len(logits), device=logits.device)
    return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2


class CurrentModelEncoder:
    """Lets the matcher use the model as it stands mid-training, remembering vectors between scores."""

    def __init__(self, model, prefixes: Tuple[str, str]):
        self.model, self.query_prefix, self.document_prefix = model, *prefixes
        self._memo: Dict[tuple, np.ndarray] = {}

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        return self.model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)

    def _remember(self, prefix: str, texts: Sequence[str]) -> np.ndarray:
        key = (prefix, *texts)
        if key not in self._memo:
            self._memo[key] = self._embed([prefix + text for text in texts])
        return self._memo[key]

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self._remember(self.query_prefix, texts)

    def encode_cached(self, texts: Sequence[str]) -> np.ndarray:
        return self._remember(self.document_prefix, texts)


def dev_mrr(model, prefixes: Tuple[str, str], diseases, queries: Dict[str, List[dict]]) -> float:
    """Mean dev MRR at the best min_similarity for the model as it stands, with no query floor."""
    encoder = CurrentModelEncoder(model, prefixes)

    def score(min_similarity: float) -> float:
        rank = matcher_ranker(SymptomMatcher(diseases, encoder, min_similarity=min_similarity, min_query_similarity=0.0))
        return float(np.mean([
            reciprocal_ranks([answer_position(rank(q["query"]), q["labels"]) for q in queries[name]]).mean()
            for name in QUERY_SETS
        ]))

    return max(score(float(value)) for value in np.arange(0.20, 0.86, 0.05))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True)
    parser.add_argument("--run", required=True, help="folder name for this experiment")
    parser.add_argument("--cross-source-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    args = parser.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer

    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    with open(ROOT / "data" / "diseases.csv", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    excluded = held_out_entries()
    pairs = training_pairs(rows, excluded, rng, args.cross_source_only)
    print(f"{len(pairs)} training entries, {sum(map(len, pairs.values()))} pairs; {len(excluded)} test entries left out", flush=True)

    short_name = Path(args.base).name
    prefixes = MODEL_PREFIXES.get(short_name, ("", ""))
    diseases = load_diseases()
    queries = {name: load_queries("dev", name) for name in QUERY_SETS}
    output = DEFAULT_CACHE_DIR / "finetuned" / args.run / short_name

    model = SentenceTransformer(args.base)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best = dev_mrr(model, prefixes, diseases, queries)
    print(f"before training: mean dev MRR {best:.4f}", flush=True)
    improved = False

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in batches(pairs, args.batch_size, rng):
            left = model.preprocess([prefixes[0] + a for a, _ in batch])
            right = model.preprocess([prefixes[1] + b for _, b in batch])
            left, right = [
                {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in features.items()}
                for features in (left, right)
            ]
            loss = contrastive_loss(model(left)["sentence_embedding"], model(right)["sentence_embedding"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        score = dev_mrr(model, prefixes, diseases, queries)
        print(f"epoch {epoch}: loss {np.mean(losses):.4f}, mean dev MRR {score:.4f}", flush=True)
        if score > best:
            best, improved = score, True
            if output.exists():
                shutil.rmtree(output)
            model.save(str(output))

    print(f"Saved the best epoch to {output}" if improved else "No epoch beat the untrained model, so nothing was saved.")


if __name__ == "__main__":
    main()
