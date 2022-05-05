from kfp import dsl

import kfp
client = kfp.Client(host='https://5fce22797569c67c-dot-europe-west1.pipelines.googleusercontent.com')

def preprocess_op():

    return dsl.ContainerOp(
        name='Preprocess Data',
        image='abouzid/gcp-project:latest',
        arguments=[],
        file_outputs={'preprocessed-dir': '/Preprocess/preprocess-data-dir.txt'}
    )
def train_op(file):

    preprocess_data_dir =(file.read())
    return dsl.ContainerOp(
        name='Train Model',
        image='abouzid/gcp-project-trainer:latest',
        arguments=['--preprocessed-dir', preprocess_data_dir],
        file_outputs={'model-dir': '/trainer/model-dir.txt'}
    )
def test_op(file_test_data, file_model):
    preprocess_data_dir =(file_test_data.read())
    model_dir =(file_model.read())
    return dsl.ContainerOp(
        name='Test Model',
        image='abouzid/gcp-project-test-model:latest',
        arguments=[
            '--preprocessed-dir', preprocess_data_dir,
            '--model-dir', model_dir
        ],
        file_outputs={
            'performance-file': '/app/output.txt'
        }
    )
@dsl.pipeline(
    name='Sentimental analyses Pipeline',
    description='An example pipeline.'
)
def Twitter_Sentimental_Pipeline():
    _preprocess_op = preprocess_op()
    _train_op = train_op(
        dsl.InputArgumentPath(_preprocess_op.outputs['preprocessed-dir'])
    ).after(_preprocess_op)
    _test_op = test_op(
        dsl.InputArgumentPath(_preprocess_op.outputs['preprocessed-dir']),
        dsl.InputArgumentPath(_train_op.outputs['model-dir'])
    ).after(_train_op)

client.create_run_from_pipeline_func(Twitter_Sentimental_Pipeline, arguments={})