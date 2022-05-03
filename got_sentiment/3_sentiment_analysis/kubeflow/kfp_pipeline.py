from kfp import dsl

import kfp
client = kfp.Client(host='https://25a7464ff82674da-dot-europe-west1.pipelines.googleusercontent.com')
def preprocess_op():

    return dsl.ContainerOp(
        name='Preprocess Data',
        image='abouzid/gcp-project:latest',
        arguments=[],
        file_outputs={}
    )
def train_op():

    return dsl.ContainerOp(
        name='Train Model',
        image='abouzid/gcp-project-trainer:latest',
        arguments=[],
        file_outputs={}
    )
@dsl.pipeline(
    name='Sentimental analyses Pipeline',
    description='An example pipeline.'
)
def boston_pipeline():
    _preprocess_op = preprocess_op()
    _train_op = train_op(
    ).after(_preprocess_op)

client.create_run_from_pipeline_func(boston_pipeline, arguments={})