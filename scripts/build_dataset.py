"""Rebuild data/diseases.csv from its sources.

Sources and terms are listed in data/SOURCES.md:
  MedlinePlus health topic summaries (public domain)
  Wikidata, "symptoms and signs" (P780) (CC0)
  Human Disease Ontology, has_symptom text (CC0)

Usage: python scripts/build_dataset.py
"""

from __future__ import annotations

import csv
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SPARQL_URL = "https://query.wikidata.org/sparql"
DOID_URL = "https://raw.githubusercontent.com/DiseaseOntology/HumanDiseaseOntology/main/src/ontology/doid.obo"
MEDLINEPLUS_INDEX_URL = "https://medlineplus.gov/xml.html"
USER_AGENT = "symptom-search-dataset-builder/1.0"
MIN_SYMPTOMS = 2
SPARQL_ATTEMPTS = 4

# Direct rdfs:label lookups, because the label service makes these queries time out.
# This also drops items with no English label. Requiring a Disease Ontology ID (P699)
# keeps to human diseases, since P780 also appears on animal diseases and even papers.
LINKS_QUERY = """
SELECT ?diseaseLabel ?symptomLabel WHERE {
  ?disease wdt:P780 ?symptom ; wdt:P699 ?doid .
  ?disease rdfs:label ?diseaseLabel . FILTER(LANG(?diseaseLabel) = "en")
  ?symptom rdfs:label ?symptomLabel . FILTER(LANG(?symptomLabel) = "en")
}
"""


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


# Entries that have "symptoms" in the source but are not conditions a person has.
_EXCLUDED_NAME = re.compile(
    r"\b(?:exposure|medicines?|drugs?|shots?|vaccines?|rehabilitation|transplant\w*|surgery|antidepressants|"
    r"tests?|screening|therapy|health|care|complications|iron|pollution|canine|feline|aleutian)\b",
    re.I,
)


def run_sparql(query: str) -> List[dict]:
    url = f"{SPARQL_URL}?format=json&query={urllib.parse.quote(query)}"
    # The endpoint can time out mid-response, which arrives as truncated JSON.
    for attempt in range(SPARQL_ATTEMPTS):
        try:
            return json.loads(fetch(url))["results"]["bindings"]
        except (json.JSONDecodeError, urllib.error.URLError):
            if attempt == SPARQL_ATTEMPTS - 1:
                raise
            time.sleep(30)
    raise AssertionError("unreachable")


def parse_disease_ontology(obo: str) -> Dict[str, Tuple[str, List[str]]]:
    """Map each DOID to its name and the symptoms named in its definition."""
    diseases = {}
    for term in obo.split("[Term]")[1:]:
        if "is_obsolete: true" in term:
            continue
        doid = re.search(r"^id: (DOID:\d+)", term, re.M)
        name = re.search(r"^name: (.+)", term, re.M)
        definition = re.search(r'^def: "(.*?)(?<!\\)"', term, re.M)
        if not (doid and name and definition):
            continue
        symptoms = re.findall(r"has_symptom ([^,.;]+?)\s*(?=,|\.|;|\bhas_symptom\b|$)", definition.group(1))
        symptoms = [re.sub(r"\s+and$", "", symptom).strip() for symptom in symptoms]
        if symptoms:
            diseases[doid.group(1)] = (name.group(1), symptoms)
    return diseases


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

    unique = dict.fromkeys(item.rstrip(".:; ") for item in items)
    return [item for item in unique if len(item) >= 3 and len(item) <= 200 and not _NOT_A_SYMPTOM.search(item)]


_PROSE_SYMPTOM = re.compile(r"\b(?:symptoms?|signs?|may have|may feel|you have|you may|you feel|people have|can cause|pain)\b", re.I)
_NOT_A_SYMPTOM = re.compile(
    r"\b(?:doctor|provider|treat\w*|diagnos\w*|tests?|surgery|medicines?|prevent\w*|risk|causes? of|caused by|"
    r"call|911|emergency|what are|not everyone|learn more|women|men|age|trigger\w*|vaccin\w*|no symptoms|"
    r"every minute|screened|children|pregnant|for example|if you do|symptoms? (?:of|can|may|usually|often|come)|"
    r"they are|this is|this can|these)\b",
    re.I,
)


def parse_medlineplus(xml_text: bytes) -> Dict[str, Tuple[str, List[str], List[str]]]:
    """Map each English topic title to its also-called names and symptoms."""
    topics = {}
    for topic in ET.fromstring(xml_text).findall("health-topic"):
        if topic.get("language") != "English":
            continue
        symptoms = parse_summary(topic.findtext("full-summary") or "")
        if symptoms:
            synonyms = [other.text for other in topic.findall("also-called") if other.text]
            topics[topic.get("title")] = (topic.get("title"), synonyms, symptoms)
    return topics


def latest_medlineplus_url() -> str:
    index = fetch(MEDLINEPLUS_INDEX_URL).decode("utf-8")
    return re.search(r"https://medlineplus\.gov/xml/mplus_topics_\d{4}-\d{2}-\d{2}\.xml", index).group(0)


def main() -> None:
    names: Dict[str, str] = {}  # normalized name -> display name
    links: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    synonyms: Dict[str, str] = {}  # normalized synonym -> normalized primary name

    # MedlinePlus goes first, so its plain-language titles become the display names.
    medlineplus_url = latest_medlineplus_url()
    for title, also_called, symptoms in parse_medlineplus(fetch(medlineplus_url)).values():
        if _EXCLUDED_NAME.search(title):
            continue
        key = normalize_name(title)
        names[key] = title
        for other in also_called:
            synonyms.setdefault(normalize_name(other), key)
        for symptom in symptoms:
            links[key][symptom].add("medlineplus")

    doid_names = parse_disease_ontology(fetch(DOID_URL).decode("utf-8"))

    def resolve(name: str) -> str:
        key = normalize_name(name)
        return synonyms.get(key, key)

    for row in run_sparql(LINKS_QUERY):
        disease, symptom = row["diseaseLabel"]["value"], row["symptomLabel"]["value"]
        if _EXCLUDED_NAME.search(disease):
            continue
        key = resolve(disease)
        names.setdefault(key, disease[0].upper() + disease[1:])
        links[key][symptom.lower()].add("wikidata")

    for name, symptoms in doid_names.values():
        if _EXCLUDED_NAME.search(name):
            continue
        key = resolve(name)
        names.setdefault(key, name[0].upper() + name[1:])
        for symptom in symptoms:
            links[key][symptom.lower()].add("disease_ontology")

    rows = []
    for key, symptoms in links.items():
        # The same symptom can differ only by case between sources.
        merged: Dict[str, Tuple[str, Set[str]]] = {}
        for symptom, sources in symptoms.items():
            display, seen = merged.get(symptom.lower(), (symptom[0].upper() + symptom[1:], set()))
            merged[symptom.lower()] = (display, seen | sources)
        if len(merged) >= MIN_SYMPTOMS:
            rows += [(names[key], display, "+".join(sorted(seen))) for display, seen in merged.values()]
    rows.sort()

    with open(DATA_DIR / "diseases.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["disease", "symptom", "sources"])
        writer.writerows(rows)

    print(f"{medlineplus_url}\n{len({r[0] for r in rows})} diseases, {len({r[1] for r in rows})} symptoms, {len(rows)} links")


if __name__ == "__main__":
    main()
