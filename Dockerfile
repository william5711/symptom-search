FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/app/.hf

# The CPU build of torch is a fraction of the size of the default CUDA build.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY symptom_search/ symptom_search/
COPY data/diseases.csv data/

# Download the model and embed the symptoms at build time, so the container
# starts without network access and the first request isn't slow.
RUN python -c "from symptom_search.api import _default_matcher; _default_matcher()"
ENV HF_HUB_OFFLINE=1

RUN useradd --create-home app && chown -R app /app
USER app

EXPOSE 8000
CMD ["uvicorn", "symptom_search.api:app", "--host", "0.0.0.0", "--port", "8000"]
