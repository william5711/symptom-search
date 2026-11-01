# Data sources

`diseases.csv` was built on 2026-11-01 by `scripts/build_dataset.py`. The `sources` column shows where each link came from. Symptom lines were extracted and filtered by script, so the file may differ from the original wording and hasn't been medically reviewed.

| Source | Used for | Terms |
|---|---|---|
| [MedlinePlus health topics](https://medlineplus.gov/xml.html), file dated 2026-10-31 | Topic names and symptom text from health topic summaries | Public domain, [per NLM](https://medlineplus.gov/about/using/usingcontent/) |
| [Wikidata](https://www.wikidata.org/), property P780 "symptoms and signs" | Symptom links on items with a Disease Ontology ID | [CC0](https://www.wikidata.org/wiki/Wikidata:Licensing) |
| [Human Disease Ontology](https://github.com/DiseaseOntology/HumanDiseaseOntology), release 2026-10-29 | Names and `has_symptom` text in definitions | [CC0](https://github.com/DiseaseOntology/HumanDiseaseOntology/blob/main/LICENSE) |

Source: MedlinePlus, National Library of Medicine. This project isn't affiliated with or endorsed by NLM, Wikidata, or the Disease Ontology project.

No content from the copyrighted A.D.A.M. Medical Encyclopedia or the ASHP drug monographs on MedlinePlus is used.

## Benchmark data

`eval/ddxplus_queries.jsonl` is adapted from the [DDXPlus Dataset](https://doi.org/10.6084/m9.figshare.20043374) (version 15) by Arsene Fansi Tchango, Rishab Goel, Zhi Wen, Julien Martel, and Joumana Ghosn, used under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changes: patients were sampled from its test file, questionnaire answers were rewritten as sentences, demographics and medical history were removed, and conditions were matched to entries in `diseases.csv`. The creators don't endorse this project.

## Model

The embedding model, [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), is downloaded at runtime and isn't part of this repo.
