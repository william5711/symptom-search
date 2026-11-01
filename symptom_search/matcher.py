"""Rank diseases by how well their known symptoms cover a patient's description.

Two signals are combined, because they fail on different queries:

* Whole text: one vector for the description, compared with the average vector of
  each disease's symptoms. Captures the overall picture.
* Clauses: each complaint is matched to its closest symptom on its own, so one
  specific complaint cannot get blurred away by the rest of the paragraph:

      score(disease) = sum over clauses of max over the disease's symptoms of
                       similarity(clause, symptom) * idf^a

  Similarities below `min_similarity` count as zero. The idf lowers symptoms that
  appear almost everywhere.

The two rankings are merged with reciprocal rank fusion, so neither signal's score
scale has to be calibrated against the other. Scores order results within one
query only. They are not probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .data import Disease
from .embeddings import cosine_matrix
from .text import is_negated, split_clauses

# Standard constant from the original reciprocal rank fusion paper (Cormack et al., 2009).
RRF_K = 60
MODES = ("hybrid", "clauses", "whole_text")


@dataclass(frozen=True)
class ClauseMatch:
    clause: str
    symptom: str
    similarity: float


@dataclass(frozen=True)
class Match:
    disease: str
    score: float
    evidence: List[ClauseMatch]


def _ranks(scores: np.ndarray) -> np.ndarray:
    """1-based rank of each entry, where the highest score gets rank 1."""
    ranks = np.empty(len(scores), dtype=int)
    ranks[np.argsort(-scores, kind="stable")] = np.arange(1, len(scores) + 1)
    return ranks


class SymptomMatcher:
    def __init__(
        self,
        diseases: Sequence[Disease],
        encoder,
        mode: str = "hybrid",
        min_similarity: float = 0.40,
        idf_power: float = 0.25,
        min_query_similarity: float = 0.55,
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

        self.diseases = list(diseases)
        self.encoder = encoder
        self.mode = mode
        self.min_similarity = min_similarity
        self.min_query_similarity = min_query_similarity
        self.symptoms = sorted({symptom for d in self.diseases for symptom in d.symptoms})

        self._symptom_vectors = encoder.encode_cached(self.symptoms)

        index = {symptom: i for i, symptom in enumerate(self.symptoms)}
        self._columns = [np.array([index[s] for s in d.symptoms]) for d in self.diseases]

        centroids = np.vstack([self._symptom_vectors[columns].mean(axis=0) for columns in self._columns])
        self._disease_vectors = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)

        document_frequency = np.bincount(np.concatenate(self._columns), minlength=len(self.symptoms))
        self._symptom_importance = (np.log((1 + len(self.diseases)) / (1 + document_frequency)) + 1) ** idf_power

    def search(self, text: str, top_k: int = 10) -> List[Match]:
        """Return the best diseases for `text`, best first. Empty if no symptom was recognised."""
        clauses = [clause for clause in split_clauses(text) if not is_negated(clause)]
        if not clauses:
            return []

        similarity = cosine_matrix(self.encoder.encode(clauses), self._symptom_vectors)
        # Text with no close symptom anywhere isn't a description of symptoms.
        if similarity.max() < self.min_query_similarity:
            return []

        relevant = similarity.max(axis=1) >= self.min_similarity
        clauses = [clause for clause, keep in zip(clauses, relevant) if keep]
        similarity = similarity[relevant]
        if not clauses:
            return []

        similarity = np.where(similarity >= self.min_similarity, similarity, 0.0)
        contribution = similarity * self._symptom_importance

        clause_scores = np.zeros(len(self.diseases))
        best_symptom = np.zeros((len(clauses), len(self.diseases)), dtype=int)
        for row, columns in enumerate(self._columns):
            candidates = contribution[:, columns]
            best_symptom[:, row] = columns[candidates.argmax(axis=1)]
            clause_scores[row] = candidates.max(axis=1).sum()

        if self.mode == "clauses":
            scores = clause_scores
        else:
            # Only recognised, non-negated clauses, so "no fever" cannot pull towards fever.
            query_vector = self.encoder.encode([". ".join(clauses)])
            whole_scores = cosine_matrix(query_vector, self._disease_vectors)[0]
            if self.mode == "whole_text":
                scores = whole_scores
            else:
                scores = 1 / (RRF_K + _ranks(clause_scores)) + 1 / (RRF_K + _ranks(whole_scores))

        matches = []
        for row in np.argsort(-scores, kind="stable")[:top_k]:
            evidence = [
                ClauseMatch(clause, self.symptoms[column], float(similarity[i, column]))
                for i, clause in enumerate(clauses)
                for column in [best_symptom[i, row]]
                if contribution[i, column] > 0
            ]
            matches.append(Match(self.diseases[row].name, float(scores[row]), evidence))
        return matches
