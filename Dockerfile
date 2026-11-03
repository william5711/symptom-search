# The Streamlit demo, as deployed to Hugging Face Spaces.
#   docker build -t symptom-search .
#   docker run -p 8501:8501 symptom-search
FROM python:3.12-slim

# Spaces run containers as user 1000, so everything the app writes must belong to it.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app

# The CPU build of torch first, so pip doesn't pull the much larger CUDA build.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.15.0 \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=user symptom_search symptom_search
COPY --chown=user data/diseases.csv data/diseases.csv
COPY --chown=user app.py .

# Download the model and embed the corpus at build time, so the first visitor doesn't wait.
RUN python -c "from symptom_search import SymptomMatcher, load_diseases; from symptom_search.embeddings import Encoder; SymptomMatcher(load_diseases(), Encoder())"

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false"]
