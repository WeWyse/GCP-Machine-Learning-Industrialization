from kfp import dsl

import kfp
client = kfp.Client(host='https://13eb97854516be9c-dot-europe-west1.pipelines.googleusercontent.com')

def preprocess_op():

    return dsl.ContainerOp(
        name='Preprocess Data',
        image='abouzid/gcp-project:latest',
        arguments=[],
        file_outputs={}
    )

@dsl.pipeline(
    name='Preproccessing Pipeline',
    description='An example pipeline.'
)
def boston_pipeline():
    _preprocess_op = preprocess_op()

client.create_run_from_pipeline_func(boston_pipeline, arguments={})