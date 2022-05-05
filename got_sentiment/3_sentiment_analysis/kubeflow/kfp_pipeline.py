from kfp import dsl
from kfp import components as comp
from kfp.components import InputPath, InputTextFile, OutputPath, OutputTextFile

import kfp
client = kfp.Client(host='https://133bb6ab465f6054-dot-europe-west1.pipelines.googleusercontent.com')

def preprocess_op():

    return dsl.ContainerOp(
        name='Preprocess Data',
        image='abouzid/gcp-project-preprocess:latest',
        arguments=[],
        file_outputs={'preprocessed-dir': '/Preprocess/preprocess-data-dir.txt'}
    )
def train_op(preprocess_data_dir : str):

    preprocess_data_dir
    return dsl.ContainerOp(
        name='Train Model',
        image='abouzid/gcp-project-trainer:latest',
        arguments=['--preprocess-data-dir', preprocess_data_dir],
        file_outputs={'model-dir': '/trainer/model-dir.txt'}
    )
def test_op(preprocess_data_dir : str , model_dir):
    return dsl.ContainerOp(
        name='Test Model',
        image='abouzid/gcp-project-test-model:latest',
        arguments=[
            '--preprocess-data-dir', preprocess_data_dir,
            '--model-dir', model_dir
        ],
        file_outputs={
            'performance-file': '/test/performance-model.txt'
        }
    )
@dsl.pipeline(
    name='Sentimental analyses Pipeline',
    description='An example pipeline.'
)
def Twitter_Sentimental_Pipeline():
    _preprocess_op = preprocess_op()
    _train_op = train_op(
        _preprocess_op.outputs['preprocessed-dir']
    ).after(_preprocess_op)
    _test_op = test_op(_preprocess_op.outputs['preprocessed-dir'],_train_op.outputs['model-dir']
    ).after(_train_op)

client.create_run_from_pipeline_func(Twitter_Sentimental_Pipeline, arguments={})