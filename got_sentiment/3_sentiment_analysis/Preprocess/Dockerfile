FROM public.ecr.aws/j1r0q0g6/notebooks/notebook-servers/jupyter-tensorflow-cuda-full:v1.4

USER root

WORKDIR /Preprocess

COPY Preprocess.py .
COPY run.py .
COPY requirements.txt .

RUN pip3 install -r requirements.txt

ENTRYPOINT [ "python", "run.py --load_json config.json" ]