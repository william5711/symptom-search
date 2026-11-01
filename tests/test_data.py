import json
from pathlib import Path

from symptom_search.data import load_diseases

QUERIES = Path(__file__).resolve().parent.parent / "eval" / "queries.jsonl"


def test_load_groups_symptoms_by_disease(tmp_path):
    path = tmp_path / "diseases.csv"
    path.write_text('disease,symptom,sources\nFlu,"Muscle, body aches",medlineplus\nFlu,Cough,wikidata\n')
    diseases = load_diseases(path)
    assert [d.name for d in diseases] == ["Flu"]
    assert diseases[0].symptoms == ("Muscle, body aches", "Cough")


def test_real_dataset_loads_cleanly():
    diseases = load_diseases()
    assert len(diseases) > 500
    assert len({d.name for d in diseases}) == len(diseases)
    assert all(d.symptoms for d in diseases)
    assert all(len(set(s.lower() for s in d.symptoms)) == len(d.symptoms) for d in diseases)


def test_benchmark_labels_exist_in_dataset():
    names = {d.name for d in load_diseases()}
    with open(QUERIES, encoding="utf-8") as file:
        labels = {label for line in file for label in json.loads(line)["labels"]}
    assert labels <= names, labels - names
