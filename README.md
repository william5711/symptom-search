# Symptom Search

You describe your symptoms in plain English, and this ranks 915 health conditions by how well their listed symptoms match. Each result shows which part of your description matched which symptom.

This is a learning project. It compares text and can't diagnose anything, so please don't use it for medical decisions.

Rewritten in 2026 with help from Claude. The original 2023 scripts are in `legacy/`.

```text
$ python -m symptom_search -k 3 "I'm thirsty all the time and pee constantly, no fever, and my vision is blurry"

1. Blood Glucose
     "I'm thirsty all the time" -> Feeling thirsty (0.84)
     "pee constantly" -> Urinating (peeing) often (0.83)
     "my vision is blurry" -> Blurred vision (0.81)
2. Diabetes Type 1
     "I'm thirsty all the time" -> Feeling very thirsty (0.84)
     "pee constantly" -> Urinating (peeing) more often, including at night (0.71)
     "my vision is blurry" -> Having blurry vision (0.88)
3. Diabetes
     "I'm thirsty all the time" -> Feeling very thirsty (0.84)
     "pee constantly" -> Urinating (peeing) more often, including at night (0.71)
     "my vision is blurry" -> Blurry vision (0.88)
```

## How it works

The input is split into clauses, and clauses that deny a symptom ("no fever") are dropped. The rest are embedded with `all-MiniLM-L6-v2`.

Conditions are then ranked two ways:

- Clause matching finds the closest listed symptom for each clause, then scores it by similarity and by how rare the symptom is across the dataset.
- Whole-text matching compares the full description with the average vector of each condition's symptoms.

Reciprocal rank fusion merges the two rankings. The scores only order results within a single query, and they aren't probabilities.

## Results

The test set is 24 patient-style descriptions that avoid naming the disease. They were written for this project and don't come from real patients. Settings were picked on a separate set of 24, and the test set was run once after those were frozen.

| Method | Hit@1 | Hit@3 | Hit@10 | MRR |
|---|---:|---:|---:|---:|
| TF-IDF keywords | 0.292 | 0.375 | 0.542 | 0.369 |
| Original 2023 method (`pooler_output`) | 0.292 | 0.500 | 0.625 | 0.421 |
| Whole text, mean pooling | 0.250 | 0.500 | 0.708 | 0.405 |
| Clauses only | 0.375 | 0.583 | 0.667 | 0.489 |
| Hybrid (default) | 0.375 | 0.583 | 0.667 | 0.494 |

The right condition is first in about 3 of 8 queries and in the top 10 in about 2 of 3, so treat the output as a shortlist.

The hybrid and clause-only versions are tied on this set, and whole-text matching does better at Hit@10. One query moves a hit rate by about 4 points, so small gaps here don't mean much. Two test queries (and four tuning queries) describe conditions the dataset doesn't have, and they count as misses.

## Data

`data/diseases.csv` has 7,010 condition-symptom links, built by `scripts/build_dataset.py` from MedlinePlus, Wikidata, and the Human Disease Ontology. See [data/SOURCES.md](data/SOURCES.md) for terms and attribution.

The data is uneven. Conditions covered by MedlinePlus have plain-language symptoms, but the median entry lists only 5 symptoms and the longest lists 73.

Other limits:

- Common conditions are missing, including gallstones, sleep apnea, sciatica, and tinnitus.
- Some entries aren't diseases. MedlinePlus has a topic called "Blood Glucose", and it can rank first.
- Symptoms were pulled from text with rules, so some lines aren't really symptoms.
- Nothing accounts for age, sex, duration, or how common a condition is.

## Known problems

- Unrelated text can still match weakly: "the weather is lovely today" returns results.
- Negation is rule-based, so "I don't have any energy" gets read as a denial and dropped.
- Two ideas didn't help during tuning and were removed: matching on Wikidata symptom aliases and normalizing by symptom-list length.

## Running it

Tested on Python 3.9. The model downloads on first run.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m symptom_search "I have a sore throat and a fever"
streamlit run app.py
python evaluate.py --split test
pytest
```

## Files

| Path | What it does |
|---|---|
| `symptom_search/data.py` | Loads the dataset |
| `symptom_search/text.py` | Splits clauses and detects negation |
| `symptom_search/embeddings.py` | Encodes text and caches vectors |
| `symptom_search/matcher.py` | Scores and ranks conditions |
| `scripts/build_dataset.py` | Rebuilds the dataset from its sources |
| `evaluate.py`, `eval/queries.jsonl` | Benchmark and labeled queries |
| `app.py` | Streamlit demo |
| `tests/` | Unit tests, which run without the model |
| `legacy/` | The original 2023 scripts, which need a dataset that is no longer in this repo |

## License

The code is under the [MIT License](LICENSE). The data has its own terms, listed in [data/SOURCES.md](data/SOURCES.md).
