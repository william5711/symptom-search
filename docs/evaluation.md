# Evaluation details

The main results are in the [README](../README.md#results). This page covers how the benchmark was built and the experiments behind the current settings.

## How the benchmark was labeled

Labels were assigned from condition names before scores for that version of the data were seen. A label can be the condition itself, a synonym or subtype, or a narrow category when no specific entry exists (cluster headache counts as "Headache"). Queries with no entry count as misses. All settings were picked on tuning queries.

The test sets have been run three times as the project changed: before the data cleanup, after it, and for the current numbers. Only combined scores were looked at, but each extra run makes the numbers a little less independent.

An earlier version of this benchmark had 24 test queries and gave a Hit@3 of 0.583. That set was too small to compare methods, and its queries are now part of the tuning set.

## What the data cleanup did

Rebuilding the data added conditions such as gallstones and tinnitus and removed topics that aren't conditions, such as "Statins" and "Blood Glucose". It helped less than the tuning results suggested. The test sets were run once before and once after, and only the combined scores were looked at:

| Hybrid, test set (`min_similarity` 0.35) | Queries with no entry | Hit@3 | Hit@10 | MRR |
|---|---:|---:|---:|---:|
| Written, before | 17 | 0.420 | 0.560 | 0.398 |
| Written, after | 6 | 0.430 | 0.570 | 0.411 |
| DDXPlus, before | 36 | 0.237 | 0.430 | 0.223 |
| DDXPlus, after | 32 | 0.220 | 0.446 | 0.221 |

The tuning queries improved more (written Hit@10 went from 0.653 to 0.755), but that gain didn't reach the test sets. These gaps are all within noise. More conditions now have an entry, but most of the newly covered queries still don't rank them near the top, so the bigger problem is ranking.

The current default `min_similarity` is 0.40, picked on the tuning queries after this comparison, which is why the README's numbers differ slightly from the "after" rows.

## Embedding models

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

## Fine-tuning

`scripts/finetune.py` trained `bge-base-en-v1.5` with a contrastive loss on pairs of symptoms from the same condition, leaving out the 161 entries used as test answers. It didn't help. Mean MRR on the tuning queries started at 0.435 with thresholds re-tuned at each epoch:

| Training pairs | Learning rate | After epoch 1 | After epoch 2 |
|---|---:|---:|---:|
| All pairs (8,514) | 2e-5 | 0.315 | 0.315 |
| Pairs across sources (2,776) | 5e-6 | 0.402 | 0.409 |

Training loss fell the whole time, so the model got better at the training task, and that task didn't match search. Two symptoms of one condition often don't mean the same thing, so pulling them together likely blurs differences that search depends on. Pairs of different wordings for the same symptom might work better, but the dataset doesn't have those labeled. No trained model is used here.

## What goes wrong

`python evaluate.py --errors` lists the hybrid's misses on the tuning queries:

| Tuning set | In the top 3 | Ranked 4 to 10 | Outside the top 10 | No entry |
|---|---:|---:|---:|---:|
| Written (98) | 61 | 14 | 19 | 4 |
| DDXPlus (95) | 26 | 16 | 37 | 16 |

Broad topics and unrelated conditions often outrank the right answer. A sprained ankle puts "Tuberculous epididymitis" in the top 3 and ranks "Sprains and Strains" 53rd, and a tension headache ranks "Headache" 74th. DDXPlus also has 16 queries for acute conditions the dataset doesn't cover.

Two ideas didn't help during tuning and were removed: matching on Wikidata symptom aliases and normalizing by symptom-list length.
