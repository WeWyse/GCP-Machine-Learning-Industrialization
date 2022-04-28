FROM python:3.7-slim

USER root

WORKDIR /trainer

COPY task.py .
COPY config.json .
COPY trainer.py .
COPY requirements.txt .

RUN pip3 install -r requirements.txt

ENTRYPOINT [ "python", "task.py --load_json config.json" ]