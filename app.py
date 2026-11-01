"""Demo UI. Run with: streamlit run app.py"""

import streamlit as st

from symptom_search import SymptomMatcher, load_diseases
from symptom_search.__main__ import DISCLAIMER
from symptom_search.embeddings import Encoder


@st.cache_resource
def load_matcher() -> SymptomMatcher:
    return SymptomMatcher(load_diseases(), Encoder())


st.set_page_config(page_title="Symptom Search")
st.title("Symptom Search")
st.caption(DISCLAIMER)

text = st.text_area(
    "Describe your symptoms in your own words",
    placeholder="I'm thirsty all the time and keep getting up at night to pee...",
)
top_k = st.slider("Results to show", min_value=3, max_value=20, value=5)

if text.strip():
    matches = load_matcher().search(text, top_k=top_k)
    if not matches:
        st.info("No known symptoms recognised. Try describing what you feel in more detail.")
    for position, match in enumerate(matches, 1):
        st.subheader(f"{position}. {match.disease}")
        if match.evidence:
            st.table(
                [
                    {"You said": item.clause, "Known symptom": item.symptom, "Similarity": f"{item.similarity:.2f}"}
                    for item in match.evidence
                ]
            )
        else:
            st.write("Matched on the overall description.")
