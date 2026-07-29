FROM python:3.11-slim
WORKDIR /srv
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
# Kronos source (provides the `model` package)
RUN git clone --depth 1 https://github.com/shiyu-coder/Kronos.git /srv/kronos
ENV PYTHONPATH=/srv/kronos
COPY requirements.txt requirements-kronos.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# Comment the next line to build a tiny naive-only image (MODE=naive)
RUN pip install --no-cache-dir -r requirements-kronos.txt --extra-index-url https://download.pytorch.org/whl/cpu
COPY app.py .
ENV MODE=auto HORIZON_DAYS=3
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
