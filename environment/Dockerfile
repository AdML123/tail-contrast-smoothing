FROM python:3.11-slim

ENV MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /code
COPY environment/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

CMD ["bash", "run.sh"]
