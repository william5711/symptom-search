# Symptom Search

You describe your symptoms in plain English, and this ranks 975 health conditions by how well their listed symptoms match. Each result shows which part of your description matched which symptom.

This is a learning project. It compares text and can't diagnose anything, so please don't use it for medical decisions.

Rewritten in 2026 with help from Claude. The original 2023 scripts are in `legacy/`.

```text
$ python -m symptom_search -k 3 "I'm thirsty all the time and pee constantly, no fever, and my vision is blurry"

1. Diabetes Type 1
     "I'm thirsty all the time" -> Feeling very thirsty (0.84)
     "pee constantly" -> Urinating (peeing) more often, including at night (0.71)
     "my vision is blurry" -> Having blurry vision (0.88)
2. Diabetes
     "I'm thirsty all the time" -> Feeling very thirsty (0.84)
     "pee constantly" -> Urinating (peeing) more often, including at night (0.71)
     "my vision is blurry" -> Blurry vision (0.88)
3. Hyperglycemia
     "I'm thirsty all the time" -> Feeling thirsty (0.84)
     "pee constantly" -> Urinating (peeing) often (0.83)
     "my vision is blurry" -> Blurred vision (0.81)
```

## How it works

The input is split into clauses, and clauses that deny a symptom ("no fever") are dropped. The rest are embedded with `all-MiniLM-L6-v2`. If no clause comes close to any listed symptom (cosine similarity below 0.55), the search returns nothing.

Conditions are then ranked two ways:

- Clause matching finds the closest listed symptom for each clause, then scores it by similarity and by how rare the symptom is across the dataset.
- Whole-text matching compares the full description with the average vector of each condition's symptoms.

Reciprocal rank fusion merges the two rankings. The scores only order results within a single query, and they aren't probabilities.

## Results

There are two held-out test sets. Brackets show 95% intervals (Wilson for Hit@3, bootstrap for MRR). Neither set comes from real patients.

**Written queries (100).** Descriptions in everyday language that avoid naming the disease, one per condition, written with Claude's help for this project. No condition appears in both these and the tuning queries. The conditions were chosen before checking whether the dataset had them, and 6 of the 100 have no entry.

| Method | Hit@1 | Hit@3 | Hit@10 | MRR |
|---|---:|---:|---:|---:|
| TF-IDF keywords | 0.200 | 0.390 [0.30, 0.49] | 0.500 | 0.306 [0.24, 0.38] |
| Original 2023 method (`pooler_output`) | 0.230 | 0.430 [0.34, 0.53] | 0.580 | 0.357 [0.28, 0.43] |
| Whole text, mean pooling | 0.320 | 0.450 [0.36, 0.55] | 0.610 | 0.418 [0.33, 0.50] |
| Clauses only | 0.270 | 0.370 [0.28, 0.47] | 0.470 | 0.335 [0.25, 0.42] |
| Hybrid (default) | 0.330 | 0.430 [0.34, 0.53] | 0.570 | 0.411 [0.33, 0.50] |

**DDXPlus patients (186).** Synthetic patients from [DDXPlus](https://doi.org/10.6084/m9.figshare.20043374), so nobody on this project wrote the cases or their diagnoses. `scripts/build_ddxplus_queries.py` turns their questionnaire answers into sentences. These read like a form, and the conditions lean toward acute care. 32 of the 186 have no entry.

| Method | Hit@1 | Hit@3 | Hit@10 | MRR |
|---|---:|---:|---:|---:|
| TF-IDF keywords | 0.032 | 0.140 [0.10, 0.20] | 0.349 | 0.129 [0.10, 0.16] |
| Original 2023 method (`pooler_output`) | 0.043 | 0.124 [0.08, 0.18] | 0.360 | 0.133 [0.10, 0.17] |
| Whole text, mean pooling | 0.059 | 0.177 [0.13, 0.24] | 0.468 | 0.170 [0.14, 0.21] |
| Clauses only | 0.086 | 0.161 [0.12, 0.22] | 0.344 | 0.161 [0.12, 0.20] |
| Hybrid (default) | 0.129 | 0.220 [0.17, 0.29] | 0.446 | 0.221 [0.18, 0.27] |

The hybrid gets the right condition into the top 3 for about 4 of 10 written queries and about 1 of 5 DDXPlus patients, so treat the output as a rough shortlist. The gap between those two numbers suggests the written queries are easier than an outside test.

The hybrid's MRR beats TF-IDF on both sets: by 0.104 [0.017, 0.189] on the written queries and by 0.092 [0.046, 0.138] on DDXPlus. On the written queries it can't be told apart from whole-text matching, and whole-text matching does slightly better at Hit@10 on both sets.

Of 30 unrelated sentences held out for testing, such as "The train to Boston leaves from platform nine", the search answered 1 and returned nothing for the other 29. All 30 used for tuning returned nothing.

An earlier version of this benchmark had 24 test queries and gave a Hit@3 of 0.583. That set was too small to compare methods, and its queries are now part of the tuning set.

Labels were assigned from condition names before scores for that version of the data were seen. A label can be the condition itself, a synonym or subtype, or a narrow category when no specific entry exists (cluster headache counts as "Headache"). Queries with no entry count as misses. The 0.55 cutoff was picked on tuning queries, and the other settings haven't changed since they were tuned on the original 24 tuning queries.

### What the data cleanup did

Rebuilding the data added conditions such as gallstones and tinnitus and removed topics that aren't conditions, such as "Statins" and "Blood Glucose". It helped less than the tuning results suggested. The test sets were run once before and once after, and only the combined scores were looked at:

| Hybrid, test set | Queries with no entry | Hit@3 | Hit@10 | MRR |
|---|---:|---:|---:|---:|
| Written, before | 17 | 0.420 | 0.560 | 0.398 |
| Written, after | 6 | 0.430 | 0.570 | 0.411 |
| DDXPlus, before | 36 | 0.237 | 0.430 | 0.223 |
| DDXPlus, after | 32 | 0.220 | 0.446 | 0.221 |

The tuning queries improved more (written Hit@10 went from 0.653 to 0.755), but that gain didn't reach the test sets. These gaps are all within noise. More conditions now have an entry, but most of the newly covered queries still don't rank them near the top, so the bigger problem is ranking.

### What goes wrong

`python evaluate.py --errors` lists the hybrid's misses on the tuning queries:

| Tuning set | In the top 3 | Ranked 4 to 10 | Outside the top 10 | No entry |
|---|---:|---:|---:|---:|
| Written (98) | 60 | 14 | 20 | 4 |
| DDXPlus (95) | 23 | 20 | 36 | 16 |

Broad topics and unrelated conditions often outrank the right answer. A sprained ankle returns "Vulvovaginal candidiasis" in the top 3, and a tension headache ranks "Headache" 82nd. DDXPlus also has 16 queries for acute conditions the dataset doesn't cover.

## Data

`data/diseases.csv` has 6,986 condition-symptom links, built by `scripts/build_dataset.py` from MedlinePlus, Wikidata, and the Human Disease Ontology. See [data/SOURCES.md](data/SOURCES.md) for terms and attribution.

The data is uneven. Conditions covered by MedlinePlus have plain-language symptoms, but the median entry lists only 5 symptoms and the longest lists 73.

Other limits:

- Common conditions are still missing, including anaphylaxis, obesity, and warts.
- Some entries are symptoms or broad categories, such as "Cough" and "Pain", and they can rank above a specific condition.
- Symptoms were pulled from text with rules, so some lines aren't really symptoms. When a MedlinePlus summary has no symptom wording, any sentence with a symptom word is used, which lets in more of these.
- Nothing accounts for age, sex, duration, or how common a condition is.

## Known problems

- The 0.55 cutoff only catches text with no close symptom at all. A real description that matches poorly can return nothing, and unrelated text that uses a body or health word can still get results.
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
