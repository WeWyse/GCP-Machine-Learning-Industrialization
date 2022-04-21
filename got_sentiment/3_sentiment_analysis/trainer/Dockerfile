FROM public.ecr.aws/j1r0q0g6/notebooks/notebook-servers/jupyter-tensorflow-cuda-full:v1.4

USER root

WORKDIR /trainer

COPY task.py .
COPY trainer.py .
COPY requirements.txt .

RUN pip3 install -r requirements.txt

ENTRYPOINT [ "python", "task.py --load_json config.json" ]