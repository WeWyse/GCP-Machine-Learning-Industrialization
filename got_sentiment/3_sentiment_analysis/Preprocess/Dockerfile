FROM python:3.7-slim

USER root

WORKDIR /Preprocess

COPY preprocess.py ./preprocess.py
COPY config.json ./config.json
COPY run.py ./run.py
COPY requirements.txt ./requirements.txt

RUN pip3 install -r requirements.txt

ENTRYPOINT [ "python", "run.py --load_json config.json" ]