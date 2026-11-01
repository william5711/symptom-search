"""Build data/diseases.csv from the saved copies in data/sources/.

This needs no network, and the same inputs always give the same file. Run
scripts/fetch_sources.py first to refresh the sources. Terms are in data/SOURCES.md.

Usage:
    python scripts/build_dataset.py            # write the file and report what changed
    python scripts/build_dataset.py --check    # fail if the file is out of date
"""

from __future__ import annotations

import argparse
import csv
import html
import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SOURCES_DIR = DATA_DIR / "sources"
MIN_SYMPTOMS = 2
MIN_BEFORE_FALLBACK = 3
MAX_ENTRY_DROP = 0.10
SOURCE_LABELS = {"medlineplus", "wikidata", "disease_ontology"}

# Entries that have "symptoms" in the source but are not conditions a person has.
_EXCLUDED_NAME = re.compile(
    r"\b(?:exposure|medicines?|drugs?|shots?|vaccines?|rehabilitation|transplant\w*|surgery|antidepressants|"
    r"tests?|screening|therapy|health|care|complications|iron|pollution|canine|feline|aleutian)\b",
    re.I,
)
# Topics the name and group checks miss: substances, habits, services, life events, and
# overviews so broad that a more specific entry is always the better answer.
_NOT_CONDITIONS = {
    "Alcohol", "Alcohol Use Disorder (AUD) Treatment", "Alzheimer's Caregivers", "Arsenic", "Asbestos", "Autoimmune Diseases",
    "Bacterial Infections", "Bereavement", "Birth Control", "Birth Weight", "Blood", "Blood Disorders", "Blood Glucose",
    "Body Weight", "Bone Density", "Botox", "Bowel Movement", "Brain Diseases", "CPR", "Calcium", "Cancer",
    "Cancer Immunotherapy", "Cancer--Living with Cancer", "Cancer in Children", "Cannabis", "Cesarean Delivery",
    "Child Sexual Abuse", "Childbirth", "Cholesterol Levels: What You Need to Know", "Cocaine",
    "Coping with Chronic Illness", "Cosmetics", "Dentures", "Diabetic Diet", "Digestive Diseases", "Elder Abuse",
    "Eye Diseases", "Eyewear", "Genetic Counseling", "Guide to Good Posture", "Healthy Sleep", "Hearing Aids",
    "Heart Disease in Women", "Heart Diseases", "Heroin", "How to Lower Cholesterol", "How to Lower Cholesterol with Diet",
    "How to Prevent Diabetes", "How to Prevent Heart Disease", "How to Prevent High Blood Pressure",
    "Immune System and Disorders", "Infectious Diseases", "Inhalants", "Intimate Partner Violence", "Kidney Diseases",
    'LDL: The "Bad" Cholesterol', "Living with HIV", "Lung Diseases", "Memory", "Menstruation", "Mental Disorders",
    "Methamphetamine", "Molds", "Neurologic Diseases", "Opioid Use Disorder (OUD) Treatment", "Orthodontia", "Ozone",
    "Pacemakers and Implantable Defibrillators", "Piercing and Tattoos", "Pregnancy", "Pregnancy and Opioids",
    "Pregnancy and Substance Use", "Quitting Smoking", "Radon", "Reproductive Hazards", "Safe Opioid Use",
    "Secondhand Smoke", "Sexual Assault", "Skin Aging", "Skin Conditions", "Smoking", "Smoking and Youth", "Sweat",
    "Tanning", "Tears", "Teen Development", "Urine and Urination", "VLDL Cholesterol", "Vascular Diseases", "Vasectomy",
    "Viral Infections", "Wounds and Injuries",
}


def normalize_name(name: str) -> str:
    """Make "Kidney Stones", "kidney stone" and "Alzheimer's Disease" compare equal."""
    name = re.sub(r"\(.*?\)", "", name.lower()).replace("'s", "")
    return re.sub(r"s$", "", re.sub(r"\s+", " ", name).strip())


def strip_tags(fragment: str) -> str:
    # Inline tags can sit inside a word ("<strong>F</strong>ace"), so they leave no space.
    fragment = re.sub(r"</?(?:a|b|strong|em|i|span)\b[^>]*>", "", fragment, flags=re.S)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<.*?>", " ", fragment, flags=re.S))).strip()


def sentences(text: str) -> List[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def parse_summary(summary: str) -> List[str]:
    """Pull symptom descriptions out of a MedlinePlus summary, which has three layouts."""
    items: List[str] = []

    # Layout 1: a heading such as "What are the symptoms of X?" and its section.
    parts = re.split(r"<h\d>(.*?)</h\d>", summary, flags=re.S)
    for heading, body in zip(parts[1::2], parts[2::2]):
        if re.search(r"symptoms?\b|\bsigns\b", heading, re.I) and not re.search(r"trigger|cause|treat", heading, re.I):
            items += [strip_tags(item) for item in re.findall(r"<li>(.*?)</li>", body, flags=re.S)]
            prose_in_section = re.sub(r"<ul>.*?</ul>", " ", body, flags=re.S)
            items += [s for s in sentences(strip_tags(prose_in_section)) if _PROSE_SYMPTOM.search(s)]

    # Layout 2: a sentence that mentions symptoms, followed by a list.
    for lead, block in re.findall(r"<p>((?:(?!</p>).)*?)</p>\s*<ul>(.*?)</ul>", summary, flags=re.S):
        if re.search(r"symptom|\bsigns?\b", lead, re.I) and not re.search(r"trigger|cause", lead, re.I):
            items += [strip_tags(item) for item in re.findall(r"<li>(.*?)</li>", block, flags=re.S)]

    # Layout 3: symptoms described in running text.
    prose = re.sub(r"<ul>.*?</ul>", " ", summary, flags=re.S)
    items += [s for s in sentences(strip_tags(prose)) if _PROSE_SYMPTOM.search(s)]

    def clean(candidates: List[str]) -> List[str]:
        unique = dict.fromkeys(item.rstrip(".:; ") for item in candidates)
        return [item for item in unique if 3 <= len(item) <= 200 and not _NOT_A_SYMPTOM.search(item)]

    found = clean(items)
    if len(found) < MIN_BEFORE_FALLBACK:
        # Layout 4: no symptom wording at all ("Hives are red and sometimes itchy bumps").
        found = clean(found + [s for s in sentences(strip_tags(prose)) if _SYMPTOM_WORD.search(s)])
    return found


_SYMPTOM_WORD = re.compile(
    r"\b(?:ach\w*|pain\w*|sore\w*|swell\w*|swollen|itch\w*|rash\w*|bumps?|blisters?|lumps?|redness|red|bleed\w*|"
    r"bruis\w*|numb\w*|tingl\w*|weak\w*|tired\w*|fatigue\w*|drows\w*|dizz\w*|lightheaded|nause\w*|vomit\w*|fevers?|"
    r"chills?|cough\w*|wheez\w*|sneez\w*|snor\w*|breathing|ringing|stools?|constipat\w*|diarrhea|cramp\w*|stiff\w*|"
    r"hoarse\w*|blurr\w*|faint\w*|confus\w*|seizures?|tremors?|sweat\w*|thirst\w*)\b",
    re.I,
)
_PROSE_SYMPTOM =re.compile(r"\b(?:symptoms?|signs?|may have|may feel|you have|you may|you feel|people have|can cause|pain)\b", re.I)
_NOT_A_SYMPTOM = re.compile(
    r"\b(?:doctor|provider|treat\w*|diagnos\w*|tests?|surgery|medicines?|prevent\w*|risk|causes? of|caused by|"
    r"call|911|emergency|what are|not everyone|learn more|women|men|age|trigger\w*|vaccin\w*|no symptoms|"
    r"every minute|screened|children|pregnant|for example|if you do|symptoms? (?:of|can|may|usually|often|come)|"
    r"they are|this is|this can|these)\b",
    re.I,
)


# A topic in one of these groups is about a service or treatment.
_SERVICE_GROUPS = {
    "Diagnostic Tests", "Drug Therapy", "Surgery and Rehabilitation", "Health System", "Disasters",
    "Transplantation and Donation", "Complementary and Alternative Therapies",
}
# These groups mix conditions with general advice, so they only rule out a topic with no other group.
_ADVICE_GROUPS = {
    "Wellness and Lifestyle", "Fitness and Exercise", "Food and Nutrition", "Safety Issues", "Population Groups",
    "Social/Family Issues", "Personal Health Issues", "Older Adults", "Children and Teenagers", "Women", "Men",
}


# Conditions filed only under an advice group.
_CONDITIONS_IN_ADVICE_GROUPS = {"Malnutrition", "Vitamin D Deficiency"}


def is_condition_topic(title: str, groups: Sequence[str]) -> bool:
    if title in _NOT_CONDITIONS:
        return False
    if title in _CONDITIONS_IN_ADVICE_GROUPS:
        return True
    return not _SERVICE_GROUPS.intersection(groups) and not set(groups) <= _ADVICE_GROUPS


def condition_topics(topics: Iterable[dict]) -> List[Tuple[str, List[str], List[str]]]:
    """Return the title, also-called names and symptoms of each condition topic."""
    conditions = []
    for topic in topics:
        if is_condition_topic(topic["title"], topic["groups"]) and not _EXCLUDED_NAME.search(topic["title"]):
            symptoms = parse_summary(topic["summary"])
            if symptoms:
                conditions.append((topic["title"], topic["also_called"], symptoms))
    return conditions


def read_csv(path: Path) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def build(sources_dir: Path = SOURCES_DIR) -> List[Tuple[str, str, str]]:
    """Merge the saved sources into sorted (disease, symptom, sources) rows."""
    names: Dict[str, str] = {}  # normalized name -> display name
    links: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    synonyms: Dict[str, str] = {}  # normalized synonym -> normalized primary name

    # MedlinePlus goes first, so its plain-language titles become the display names.
    with open(sources_dir / "medlineplus_topics.jsonl", encoding="utf-8") as file:
        topics = [json.loads(line) for line in file]
    for title, also_called, symptoms in condition_topics(topics):
        key = normalize_name(title)
        names[key] = title
        for other in also_called:
            synonyms.setdefault(normalize_name(other), key)
        for symptom in symptoms:
            links[key][symptom].add("medlineplus")

    # Wikidata and the Disease Ontology use some of the same overview names.
    excluded_keys = {normalize_name(title) for title in _NOT_CONDITIONS}

    def add_link(disease: str, symptom: str, source: str) -> None:
        if _EXCLUDED_NAME.search(disease) or normalize_name(disease) in excluded_keys:
            return
        key = normalize_name(disease)
        key = synonyms.get(key, key)
        names.setdefault(key, disease[0].upper() + disease[1:])
        links[key][symptom.lower()].add(source)

    for row in read_csv(sources_dir / "wikidata_links.csv"):
        add_link(row["disease"], row["symptom"], "wikidata")
    for row in read_csv(sources_dir / "disease_ontology_symptoms.csv"):
        add_link(row["disease"], row["symptom"], "disease_ontology")

    rows = []
    for key, symptoms in links.items():
        # The same symptom can differ only by case between sources.
        merged: Dict[str, Tuple[str, Set[str]]] = {}
        for symptom, sources in symptoms.items():
            display, seen = merged.get(symptom.lower(), (symptom[0].upper() + symptom[1:], set()))
            merged[symptom.lower()] = (display, seen | sources)
        # One MedlinePlus sentence can cover several symptoms, unlike one Wikidata link.
        from_medlineplus = any("medlineplus" in seen for _, seen in merged.values())
        if len(merged) >= MIN_SYMPTOMS or from_medlineplus:
            rows += [(names[key], display, "+".join(sorted(seen))) for display, seen in merged.values()]
    return sorted(rows)


def validate(rows: Sequence[Tuple[str, str, str]], previous_entries: int = 0) -> List[str]:
    """Return every problem found. A rebuild with problems must not replace the dataset."""
    problems = []
    if any(not (disease.strip() and symptom.strip()) for disease, symptom, _ in rows):
        problems.append("a row has an empty disease or symptom")
    pairs = [(disease, symptom.lower()) for disease, symptom, _ in rows]
    if len(pairs) != len(set(pairs)):
        problems.append("a symptom is listed twice for one disease")
    unknown = {label for _, _, sources in rows for label in sources.split("+")} - SOURCE_LABELS
    if unknown:
        problems.append(f"unknown sources: {sorted(unknown)}")
    entries = {disease for disease, _, _ in rows}
    excluded = sorted(entries & _NOT_CONDITIONS)
    if excluded:
        problems.append(f"entries that aren't conditions: {excluded}")
    if previous_entries and len(entries) < previous_entries * (1 - MAX_ENTRY_DROP):
        problems.append(f"entries fell from {previous_entries} to {len(entries)}, which suggests a broken source")
    return problems


def to_csv(rows: Sequence[Tuple[str, str, str]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(["disease", "symptom", "sources"])
    writer.writerows(rows)
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="fail if data/diseases.csv doesn't match its sources")
    args = parser.parse_args()

    path = DATA_DIR / "diseases.csv"
    old_rows = read_csv(path) if path.exists() else []
    old_entries = {row["disease"] for row in old_rows}
    rows = build()
    new_entries = {disease for disease, _, _ in rows}

    problems = validate(rows, len(old_entries))
    if problems:
        sys.exit("Dataset not written:\n  " + "\n  ".join(problems))

    if args.check:
        current = path.read_bytes().decode("utf-8") if path.exists() else ""
        sys.exit(0 if current == to_csv(rows) else "data/diseases.csv doesn't match data/sources/. Run scripts/build_dataset.py.")

    with open(path, "w", newline="", encoding="utf-8") as file:
        file.write(to_csv(rows))
    print(f"{len(new_entries)} entries, {len({r[1] for r in rows})} symptoms, {len(rows)} links")
    print(f"Since the last build: {len(new_entries - old_entries)} entries added, {len(old_entries - new_entries)} removed, "
          f"{len(rows) - len(old_rows):+d} links")
    for label, names in (("Added", new_entries - old_entries), ("Removed", old_entries - new_entries)):
        if names:
            print(f"{label}: {', '.join(sorted(names))}")


if __name__ == "__main__":
    main()
