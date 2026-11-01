"""Download the three sources and save what the dataset needs in data/sources/.

The sources change over time, so scripts/build_dataset.py works only from these saved
copies. manifest.json records where each came from, when, and its checksum.

Usage: python scripts/fetch_sources.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path
from typing import List, Tuple

SOURCES_DIR = Path(__file__).resolve().parent.parent / "data" / "sources"
SPARQL_URL = "https://query.wikidata.org/sparql"
DOID_URL = "https://raw.githubusercontent.com/DiseaseOntology/HumanDiseaseOntology/main/src/ontology/doid.obo"
MEDLINEPLUS_INDEX_URL = "https://medlineplus.gov/xml.html"
USER_AGENT = "symptom-search-dataset-builder/1.0"
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


def parse_disease_ontology(obo: str) -> List[Tuple[str, str, str]]:
    """Return (DOID, name, symptom) for each symptom named in a term's definition."""
    rows = []
    for term in obo.split("[Term]")[1:]:
        if "is_obsolete: true" in term:
            continue
        doid = re.search(r"^id: (DOID:\d+)", term, re.M)
        name = re.search(r"^name: (.+)", term, re.M)
        definition = re.search(r'^def: "(.*?)(?<!\\)"', term, re.M)
        if not (doid and name and definition):
            continue
        symptoms = re.findall(r"has_symptom ([^,.;]+?)\s*(?=,|\.|;|\bhas_symptom\b|$)", definition.group(1))
        rows += [(doid.group(1), name.group(1), re.sub(r"\s+and$", "", symptom).strip()) for symptom in symptoms]
    return rows


def parse_medlineplus_topics(xml_text: bytes) -> List[dict]:
    topics = []
    for topic in ET.fromstring(xml_text).findall("health-topic"):
        if topic.get("language") == "English":
            topics.append({
                "title": topic.get("title"),
                "groups": sorted(group.text for group in topic.findall("group")),
                "also_called": sorted(other.text for other in topic.findall("also-called") if other.text),
                "summary": topic.findtext("full-summary") or "",
            })
    return sorted(topics, key=lambda topic: topic["title"])


def latest_medlineplus_url() -> str:
    index = fetch(MEDLINEPLUS_INDEX_URL).decode("utf-8")
    return re.search(r"https://medlineplus\.gov/xml/mplus_topics_\d{4}-\d{2}-\d{2}\.xml", index).group(0)


def write_csv(path: Path, header: List[str], rows: List[Tuple[str, ...]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(sorted(set(rows)))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    SOURCES_DIR.mkdir(parents=True, exist_ok=True)

    medlineplus_url = latest_medlineplus_url()
    topics = parse_medlineplus_topics(fetch(medlineplus_url))
    medlineplus_path = SOURCES_DIR / "medlineplus_topics.jsonl"
    with open(medlineplus_path, "w", encoding="utf-8") as file:
        for topic in topics:
            file.write(json.dumps(topic, ensure_ascii=False) + "\n")

    wikidata_rows = [(row["diseaseLabel"]["value"], row["symptomLabel"]["value"]) for row in run_sparql(LINKS_QUERY)]
    wikidata_path = SOURCES_DIR / "wikidata_links.csv"
    write_csv(wikidata_path, ["disease", "symptom"], wikidata_rows)

    ontology_rows = parse_disease_ontology(fetch(DOID_URL).decode("utf-8"))
    ontology_path = SOURCES_DIR / "disease_ontology_symptoms.csv"
    write_csv(ontology_path, ["doid", "disease", "symptom"], ontology_rows)

    today = date.today().isoformat()
    manifest = {
        path.name: {"url": url, "retrieved": today, "records": records, "sha256": sha256(path)}
        for path, url, records in [
            (medlineplus_path, medlineplus_url, len(topics)),
            (wikidata_path, SPARQL_URL, len(set(wikidata_rows))),
            (ontology_path, DOID_URL, len(set(ontology_rows))),
        ]
    }
    (SOURCES_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for name, entry in manifest.items():
        print(f"{name}: {entry['records']} records")


if __name__ == "__main__":
    main()
