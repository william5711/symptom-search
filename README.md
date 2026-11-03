# Symptom Search

<p align="center">
  <img src="docs/demo.png" alt="Symptom Search ranking Diabetes Type 1, Diabetes and Hyperglycemia for a description of thirst, frequent urination and blurry vision, with the matched symptom for each phrase" width="560">
  <br><em>The Streamlit demo (<code>streamlit run app.py</code>) with the example query below.</em>
</p>

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
| Whole text, mean pooling | 0.330 | 0.450 [0.36, 0.55] | 0.600 | 0.420 [0.34, 0.50] |
| Clauses only | 0.250 | 0.360 [0.27, 0.46] | 0.430 | 0.319 [0.24, 0.40] |
| Hybrid (default) | 0.320 | 0.420 [0.33, 0.52] | 0.560 | 0.398 [0.32, 0.48] |

**DDXPlus patients (186).** Synthetic patients from [DDXPlus](https://doi.org/10.6084/m9.figshare.20043374), so nobody on this project wrote the cases or their diagnoses. `scripts/build_ddxplus_queries.py` turns their questionnaire answers into sentences. These read like a form, and the conditions lean toward acute care. 32 of the 186 have no entry.

| Method | Hit@1 | Hit@3 | Hit@10 | MRR |
|---|---:|---:|---:|---:|
| TF-IDF keywords | 0.032 | 0.140 [0.10, 0.20] | 0.349 | 0.129 [0.10, 0.16] |
| Original 2023 method (`pooler_output`) | 0.043 | 0.124 [0.08, 0.18] | 0.360 | 0.133 [0.10, 0.17] |
| Whole text, mean pooling | 0.059 | 0.188 [0.14, 0.25] | 0.473 | 0.175 [0.14, 0.21] |
| Clauses only | 0.086 | 0.161 [0.12, 0.22] | 0.306 | 0.159 [0.12, 0.20] |
| Hybrid (default) | 0.124 | 0.226 [0.17, 0.29] | 0.441 | 0.213 [0.17, 0.26] |

The hybrid gets the right condition into the top 3 for about 4 of 10 written queries and fewer than 1 of 4 DDXPlus patients, so treat the output as a rough shortlist. The gap between those two numbers suggests the written queries are easier than an outside test.

The hybrid's MRR beats TF-IDF on both sets: by 0.091 [0.007, 0.175] on the written queries and by 0.084 [0.037, 0.129] on DDXPlus. Whole-text matching scored higher on the written queries (MRR 0.420 vs 0.398) and at Hit@10 on both sets, while the hybrid was ahead on DDXPlus MRR.

Of 30 unrelated sentences held out for testing, such as "The train to Boston leaves from platform nine", the search answered 1 and returned nothing for the other 29. All 30 used for tuning returned nothing. The cutoff has a cost: it refused 3 of the 100 written test queries, and without it the hybrid's written MRR would be 0.418.

An earlier version of this benchmark had 24 test queries and gave a Hit@3 of 0.583. That set was too small to compare methods, and its queries are now part of the tuning set.

Labels were assigned from condition names before scores for that version of the data were seen. A label can be the condition itself, a synonym or subtype, or a narrow category when no specific entry exists (cluster headache counts as "Headache"). Queries with no entry count as misses. All settings were picked on tuning queries. The test sets have been run three times as the project changed: before the data cleanup, after it, and for the current numbers. Only combined scores were looked at, but each extra run makes the numbers a little less independent.

### What the data cleanup did

Rebuilding the data added conditions such as gallstones and tinnitus and removed topics that aren't conditions, such as "Statins" and "Blood Glucose". It helped less than the tuning results suggested. The test sets were run once before and once after, and only the combined scores were looked at:

| Hybrid, test set (`min_similarity` 0.35) | Queries with no entry | Hit@3 | Hit@10 | MRR |
|---|---:|---:|---:|---:|
| Written, before | 17 | 0.420 | 0.560 | 0.398 |
| Written, after | 6 | 0.430 | 0.570 | 0.411 |
| DDXPlus, before | 36 | 0.237 | 0.430 | 0.223 |
| DDXPlus, after | 32 | 0.220 | 0.446 | 0.221 |

The tuning queries improved more (written Hit@10 went from 0.653 to 0.755), but that gain didn't reach the test sets. These gaps are all within noise. More conditions now have an entry, but most of the newly covered queries still don't rank them near the top, so the bigger problem is ranking.

### Embedding models

`scripts/compare_models.py` runs nine models in the hybrid. Each model's thresholds were tuned on the tuning queries, because similarity scales differ between models. Test results:

| Model | Written Hit@3 | Written MRR | DDXPlus Hit@3 | DDXPlus MRR | Combined MRR gain over MiniLM | Corrected p |
|---|---:|---:|---:|---:|---:|---:|
| `all-MiniLM-L6-v2` (default) | 0.420 | 0.398 | 0.226 | 0.213 | | |
| `bge-base-en-v1.5` | 0.530 | 0.466 | 0.274 | 0.227 | +0.033 [0.006, 0.061] | 0.10 |
| `all-mpnet-base-v2` | 0.480 | 0.429 | 0.247 | 0.202 | +0.004 [-0.023, 0.030] | 1.00 |
| `bge-small-en-v1.5` | 0.440 | 0.379 | 0.258 | 0.217 | -0.004 [-0.029, 0.021] | 1.00 |
| PubMedBERT (`pubmedbert-base-embeddings`) | 0.330 | 0.267 | 0.231 | 0.217 | -0.043 [-0.082, -0.003] | 0.10 |
| SapBERT | 0.330 | 0.279 | 0.237 | 0.203 | -0.048 [-0.085, -0.010] | 0.06 |
| `e5-base-v2` | 0.420 | 0.375 | 0.156 | 0.149 | -0.050 [-0.084, -0.016] | 0.03 |
| `e5-small-v2` | 0.360 | 0.315 | 0.129 | 0.143 | -0.075 [-0.110, -0.042] | <0.01 |
| BioBERT | 0.020 | 0.016 | 0.005 | 0.007 | -0.267 [-0.313, -0.225] | <0.01 |

The p-values come from a paired sign-flip test on both sets together, with a Holm correction for testing eight models. Before the test run, the rule was to switch the default to `bge-base-en-v1.5` only if its corrected p was below 0.05. It wasn't, so MiniLM stays, and it's also about a fifth of the download size. `bge-base-en-v1.5` is still the one to try next, since its uncorrected p is 0.025 and it needs more test data to settle.

The medical models did worst or near worst on the written queries, which use everyday wording. BioBERT isn't built for sentence similarity, and its scores suffer further because its similarities are so high that the unrelated-text cutoff refuses most queries.

### Fine-tuning

`scripts/finetune.py` trained `bge-base-en-v1.5` with a contrastive loss on pairs of symptoms from the same condition, leaving out the 161 entries used as test answers. It didn't help. Mean MRR on the tuning queries started at 0.435 with thresholds re-tuned at each epoch:

| Training pairs | Learning rate | After epoch 1 | After epoch 2 |
|---|---:|---:|---:|
| All pairs (8,514) | 2e-5 | 0.315 | 0.315 |
| Pairs across sources (2,776) | 5e-6 | 0.402 | 0.409 |

Training loss fell the whole time, so the model got better at the training task, and that task didn't match search. Two symptoms of one condition often don't mean the same thing, so pulling them together likely blurs differences that search depends on. Pairs of different wordings for the same symptom might work better, but the dataset doesn't have those labeled. No trained model is used here.

### What goes wrong

`python evaluate.py --errors` lists the hybrid's misses on the tuning queries:

| Tuning set | In the top 3 | Ranked 4 to 10 | Outside the top 10 | No entry |
|---|---:|---:|---:|---:|
| Written (98) | 61 | 14 | 19 | 4 |
| DDXPlus (95) | 26 | 16 | 37 | 16 |

Broad topics and unrelated conditions often outrank the right answer. A sprained ankle puts "Tuberculous epididymitis" in the top 3 and ranks "Sprains and Strains" 53rd, and a tension headache ranks "Headache" 74th. DDXPlus also has 16 queries for acute conditions the dataset doesn't cover.

## Data

`data/diseases.csv` has 6,986 condition-symptom links from MedlinePlus, Wikidata, and the Human Disease Ontology. See [data/SOURCES.md](data/SOURCES.md) for terms and attribution.

The sources change over time, so the build has two steps:

- `scripts/fetch_sources.py` downloads the sources, saves what's needed in `data/sources/`, and records each one's URL, date, and checksum in `manifest.json`.
- `scripts/build_dataset.py` builds the dataset from those saved copies with no network access, so the same inputs always give the same file. It refuses to write the result if a row is empty or duplicated, a source is unknown, a non-condition slips in, or the entry count falls by more than 10%.

GitHub Actions runs the tests on every push, including a check that the dataset matches its saved sources. A monthly job reports how far the live sources have drifted. It doesn't commit anything, because a data change also means rerunning the benchmark.

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

Tested on Python 3.12, with exact versions pinned in `requirements.txt`. The model downloads on first run.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m symptom_search "I have a sore throat and a fever"
streamlit run app.py
python evaluate.py --split test
python evaluate.py --split test --queries ddxplus
python scripts/compare_models.py --split test
pytest
```

## Files

| Path | What it does |
|---|---|
| `symptom_search/data.py` | Loads the dataset |
| `symptom_search/text.py` | Splits clauses and detects negation |
| `symptom_search/embeddings.py` | Encodes text and caches vectors |
| `symptom_search/matcher.py` | Scores and ranks conditions |
| `scripts/fetch_sources.py` | Downloads the sources into `data/sources/` |
| `scripts/build_dataset.py` | Builds and validates the dataset from the saved sources |
| `scripts/compare_models.py`, `eval/model_settings.json` | Compares embedding models and stores their tuned thresholds |
| `scripts/finetune.py` | The fine-tuning experiment |
| `.github/workflows/` | Tests on every push and the monthly source check |
| `evaluate.py`, `eval/` | Benchmark and labeled queries |
| `scripts/build_ddxplus_queries.py` | Samples DDXPlus patients into benchmark queries |
| `app.py` | Streamlit demo |
| `tests/` | Unit tests, which run without the model |
| `legacy/` | The original 2023 scripts, which need a dataset that is no longer in this repo |

## License

The code is under the [MIT License](LICENSE). The data has its own terms, listed in [data/SOURCES.md](data/SOURCES.md).
