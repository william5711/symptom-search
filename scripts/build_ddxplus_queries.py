"""Sample DDXPlus synthetic patients into eval/ddxplus_queries.jsonl.

DDXPlus is an outside check on the benchmark: neither its patients nor their
diagnoses were written for this project. Get the English files from
https://doi.org/10.6084/m9.figshare.20043374 and pass their folder:

    python scripts/build_ddxplus_queries.py path/to/ddxplus

Answers to the questionnaire become plain sentences. Antecedents such as "Do
you have COPD?" are left out because they can name the diagnosis.
"""

from __future__ import annotations

import ast
import csv
import json
import random
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

OUTPUT = Path(__file__).resolve().parent.parent / "eval" / "ddxplus_queries.jsonl"
SEED = 20261101
DEV_PER_CONDITION = 2
TEST_PER_CONDITION = 4
MIN_SENTENCES = 2
PAIN_LOCATION = "E_55"

# Matched by condition name alone, before any scores were seen. An empty list
# means the dataset has no entry, and the patient counts as a miss.
LABELS = {
    "Anaphylaxis": [],
    "Unstable angina": ["Angina"],
    "Stable angina": ["Angina"],
    "Anemia": ["Anemia"],
    "Bronchospasm / acute asthma exacerbation": ["Asthma", "Allergic asthma", "Intrinsic asthma"],
    "Panic attack": ["Panic Disorder"],
    "Bronchiectasis": [],
    "Bronchiolitis": ["Bronchiolitis"],
    "Bronchitis": ["Acute Bronchitis", "Bronchitis"],
    "Chagas": ["Chagas Disease"],
    "Whooping cough": ["Whooping Cough", "Bordetella parapertussis whooping cough"],
    "Cluster headache": ["Headache"],
    "Ebola": ["Ebola"],
    "Pulmonary embolism": ["Pulmonary Embolism"],
    "Acute COPD exacerbation / infection": ["COPD", "Chronic obstructive pulmonary disease"],
    "Atrial fibrillation": ["Atrial Fibrillation"],
    "Spontaneous rib fracture": ["Fractures"],
    "Inguinal hernia": ["Hernia"],
    "URTI": ["Common Cold", "Nasopharyngitis"],
    "Acute laryngitis": ["Voice Disorders"],
    "Croup": ["Croup"],
    "Larygospasm": [],
    "SLE": ["Lupus", "Lupus erythematosus"],
    "Myasthenia gravis": ["Myasthenia Gravis"],
    "Myocarditis": [],
    "Pancreatic neoplasm": ["Pancreatic Cancer"],
    "Acute pulmonary edema": [],
    "Localized edema": [],
    "Acute otitis media": ["Otitis media"],
    "Viral pharyngitis": ["Nasopharyngitis", "Sore Throat"],
    "Pneumonia": ["Pneumonia"],
    "Spontaneous pneumothorax": ["Pneumothorax", "Collapsed Lung"],
    "Possible NSTEMI / STEMI": ["Heart Attack"],
    "Influenza": ["Flu"],
    "Pericarditis": ["Pericarditis"],
    "GERD": ["GERD"],
    "Allergic sinusitis": ["Allergic rhinitis", "Hay Fever"],
    "Acute rhinosinusitis": ["Sinusitis"],
    "Chronic rhinosinusitis": ["Sinusitis"],
    "Acute dystonic reactions": [],
    "Sarcoidosis": ["Sarcoidosis"],
    "Scombroid food poisoning": [],
    "Boerhaave": ["Esophagus Disorders"],
    "Guillain-Barré syndrome": ["Guillain-Barre Syndrome"],
    "PSVT": ["Arrhythmia"],
    "Tuberculosis": ["Tuberculosis", "Pulmonary tuberculosis"],
    "HIV (initial infection)": ["HIV", "HIV/AIDS"],
    "Pulmonary neoplasm": ["Lung Cancer", "Lung carcinoma", "Non-small-cell lung carcinoma"],
    "Epiglottitis": [],
}

# Questions that say nothing on their own, or that name the diagnosis.
SKIPPED = {"E_13", "E_53", "E_129", "E_150", "E_175", "E_202"}

# Questions the prefix rules below can't turn into a sentence.
SENTENCES = {
    "E_23": "I temporarily stop breathing while I'm asleep.",
    "E_92": "My cheeks suddenly turned red.",
    "E_159": "I lost consciousness.",
    "E_166": "I vomited after coughing.",
    "E_176": "I have had weakness or paralysis in one or more of my limbs or in my face.",
    "E_177": "I have had numbness, loss of sensitivity or tingling on my body.",
    "E_205": "I suddenly have difficulty opening my mouth or have jaw pain when opening it.",
    "E_67": "I have bouts of choking or shortness of breath that wake me up at night.",
    "E_112": "I wheeze while inhaling or my breathing is noisy after coughing spells.",
    "E_128": "I have felt like I was suffocating for a very short time and could not breathe or speak.",
    "E_148": "I am feeling nauseous or feel like vomiting.",
    "E_161": "I have recently had a loss of appetite or get full more quickly than usual.",
    "E_192": "Muscle spasms or soreness in my neck are keeping me from turning my head to one side.",
    "E_211": "I have vomited several times or made several efforts to vomit.",
    "E_75": "I feel like I am choking or suffocating.",
    "E_82": "I feel lightheaded and dizzy, or like I am about to faint.",
    "E_88": "I feel so tired that I am unable to do my usual activities or I am stuck in my bed all day long.",
    "E_89": "I constantly feel fatigued or have non-restful sleep.",
    "E_111": "I feel like I am dying or was afraid that I was about to die.",
    "E_114": "I am more irritable or my mood has been very unstable recently.",
    "E_154": "My skin is much paler than usual.",
    "E_155": "I feel my heart beating fast or irregularly, or I feel palpitations.",
    "E_169": "My nose or the back of my throat is itchy.",
    "E_174": "I have been unintentionally losing weight or have lost my appetite.",
    "E_217": "My symptoms are worse when lying down and alleviated while sitting up.",
    "E_219": "My symptoms are more prominent at night.",
    "E_221": "My symptoms or pain increase with coughing, lifting a weight or forcing a bowel movement.",
}

_PREFIXES = [
    ("Do you ", "I "),
    ("Have you ", "I have "),
    ("Are you ", "I am "),
    ("Do your ", "My "),
]
_LOCATION_FIXES = {"palace": "palate", "ischio jambier": "hamstring"}


def to_sentence(question: str) -> str:
    """Turn a yes/no questionnaire item into the first-person statement it confirms."""
    for prefix, replacement in _PREFIXES:
        if question.startswith(prefix):
            body = question[len(prefix):].rstrip("?").strip()
            body = re.sub(r"\byou are\b", "I am", body)
            body = re.sub(r"\byou're\b", "I'm", body)
            body = re.sub(r"\byour\b", "my", body)
            body = re.sub(r"\byou\b", "I", body)
            return f"{replacement}{body}."
    raise ValueError(f"No sentence rule for: {question}")


def location_phrase(meaning: str) -> str:
    match = re.match(r"(.*)\((L|R)\)$", meaning)
    if match:
        side = "left" if match.group(2) == "L" else "right"
        return f"{side} {match.group(1).strip()}"
    return _LOCATION_FIXES.get(meaning, meaning)


def describe(evidences: List[str], catalog: Dict[str, dict]) -> str:
    """Build a description from a patient's symptoms and pain locations."""
    sentences, locations = [], []
    for item in evidences:
        name, _, value = item.partition("_@_")
        evidence = catalog[name]
        if evidence["is_antecedent"] or name in SKIPPED:
            continue
        if name == PAIN_LOCATION:
            meaning = evidence["value_meaning"][value]["en"]
            if meaning != "nowhere":
                locations.append(location_phrase(meaning))
        elif evidence["data_type"] == "B":
            sentences.append(SENTENCES.get(name) or to_sentence(evidence["question_en"]))
    if locations:
        sentences.insert(0, f"I have pain in my {', '.join(locations)}.")
    return " ".join(sentences)


def sample(folder: Path) -> List[dict]:
    catalog = json.loads((folder / "release_evidences.json").read_text(encoding="utf-8"))
    conditions = json.loads((folder / "release_conditions.json").read_text(encoding="utf-8"))
    english = {name: c["cond-name-eng"] for name, c in conditions.items()}
    missing = set(english.values()) ^ set(LABELS)
    if missing:
        raise ValueError(f"Labels and conditions differ: {sorted(missing)}")

    by_condition = defaultdict(list)
    with zipfile.ZipFile(folder / "release_test_patients.zip") as archive:
        with archive.open("release_test_patients") as file:
            for row in csv.DictReader(line.decode("utf-8") for line in file):
                by_condition[row["PATHOLOGY"]].append(row["EVIDENCES"])

    rng = random.Random(SEED)
    queries = []
    seen = set()
    for pathology in sorted(by_condition):
        patients = sorted(by_condition[pathology])
        rng.shuffle(patients)
        chosen: List[str] = []
        for patient in patients:
            text = describe(ast.literal_eval(patient), catalog)
            if len(re.findall(r"\.(?: |$)", text)) >= MIN_SENTENCES and text not in seen:
                seen.add(text)
                chosen.append(text)
            if len(chosen) == DEV_PER_CONDITION + TEST_PER_CONDITION:
                break
        condition = english.get(pathology) or pathology
        for i, text in enumerate(chosen):
            queries.append({
                "split": "dev" if i < DEV_PER_CONDITION else "test",
                "condition": condition,
                "labels": LABELS[condition],
                "query": text,
            })
    return queries


def main(folder: Optional[str] = None) -> None:
    if folder is None:
        sys.exit(__doc__)
    queries = sample(Path(folder))
    with open(OUTPUT, "w", encoding="utf-8") as file:
        for query in queries:
            file.write(json.dumps(query, ensure_ascii=False) + "\n")
    for split in ("dev", "test"):
        part = [q for q in queries if q["split"] == split]
        print(f"{split}: {len(part)} queries, {sum(not q['labels'] for q in part)} with no entry")


if __name__ == "__main__":
    main(*sys.argv[1:2])
