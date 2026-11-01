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

There are two held-out test sets. Brackets show 95% intervals (Wilson for Hit@3, bootstrap for MRR). Neither set comes from real patients.

**Written queries (100).** Descriptions in everyday language that avoid naming the disease, one per condition, written with Claude's help for this project. No condition appears in both these and the tuning queries. The conditions were chosen before checking whether the dataset had them, and 17 of the 100 have no entry.

| Method | Hit@1 | Hit@3 | Hit@10 | MRR |
|---|---:|---:|---:|---:|
| TF-IDF keywords | 0.180 | 0.310 [0.23, 0.41] | 0.440 | 0.267 [0.20, 0.34] |
| Original 2023 method (`pooler_output`) | 0.210 | 0.370 [0.28, 0.47] | 0.510 | 0.313 [0.24, 0.39] |
| Whole text, mean pooling | 0.300 | 0.430 [0.34, 0.53] | 0.570 | 0.390 [0.31, 0.48] |
| Clauses only | 0.240 | 0.350 [0.26, 0.45] | 0.460 | 0.314 [0.24, 0.40] |
| Hybrid (default) | 0.320 | 0.420 [0.33, 0.52] | 0.560 | 0.398 [0.31, 0.49] |

**DDXPlus patients (186).** Synthetic patients from [DDXPlus](https://doi.org/10.6084/m9.figshare.20043374), so nobody on this project wrote the cases or their diagnoses. `scripts/build_ddxplus_queries.py` turns their questionnaire answers into sentences. These read like a form, and the conditions lean toward acute care. 36 of the 186 have no entry.

| Method | Hit@1 | Hit@3 | Hit@10 | MRR |
|---|---:|---:|---:|---:|
| TF-IDF keywords | 0.027 | 0.129 [0.09, 0.18] | 0.317 | 0.120 [0.09, 0.15] |
| Original 2023 method (`pooler_output`) | 0.038 | 0.118 [0.08, 0.17] | 0.371 | 0.132 [0.10, 0.16] |
| Whole text, mean pooling | 0.043 | 0.129 [0.09, 0.18] | 0.430 | 0.148 [0.12, 0.18] |
| Clauses only | 0.097 | 0.172 [0.12, 0.23] | 0.360 | 0.172 [0.13, 0.22] |
| Hybrid (default) | 0.134 | 0.237 [0.18, 0.30] | 0.430 | 0.223 [0.18, 0.27] |

The hybrid gets the right condition into the top 3 for about 4 of 10 written queries and about 1 of 4 DDXPlus patients, so treat the output as a rough shortlist. The gap between those two numbers suggests the written queries are easier than an outside test.

The hybrid's MRR beats TF-IDF on both sets: by 0.131 [0.050, 0.212] on the written queries and by 0.103 [0.057, 0.149] on DDXPlus. On the written queries it can't be told apart from whole-text matching.

An earlier version of this benchmark had 24 test queries and gave a Hit@3 of 0.583. That set was too small to compare methods, and its queries are now part of the tuning set.

Labels were assigned from condition names before any scores were seen. A label can be the condition itself, a synonym or subtype, or a narrow category when no specific entry exists (cluster headache counts as "Headache"). Queries with no entry count as misses. Settings have not changed since they were tuned on the original 24 tuning queries.

### What goes wrong

`python evaluate.py --errors` lists the hybrid's misses on the tuning queries:

| Tuning set | In the top 3 | Ranked 4 to 10 | Outside the top 10 | No entry |
|---|---:|---:|---:|---:|
| Written (98) | 55 | 9 | 20 | 14 |
| DDXPlus (95) | 23 | 18 | 36 | 18 |

Missing entries alone cost 14% to 19% of queries. Among the rest, broad topics and rare diseases often outrank the right answer: a sprained ankle returns "Vulvovaginal candidiasis" in the top 3, and a tension headache ranks "Heart Disease in Women" above "Headache". Distinctive wording also gets lost, as with the barking cough in croup.

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
python evaluate.py --split test --queries ddxplus
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
| `evaluate.py`, `eval/` | Benchmark and labeled queries |
| `scripts/build_ddxplus_queries.py` | Samples DDXPlus patients into benchmark queries |
| `app.py` | Streamlit demo |
| `tests/` | Unit tests, which run without the model |
| `legacy/` | The original 2023 scripts, which need a dataset that is no longer in this repo |

## License

The code is under the [MIT License](LICENSE). The data has its own terms, listed in [data/SOURCES.md](data/SOURCES.md).
