from kfp import dsl
from kfp import components as comp
from kfp.components import InputPath, InputTextFile, OutputPath, OutputTextFile

import kfp
client = kfp.Client(host='https://b7d1f1b3cbef594-dot-europe-west1.pipelines.googleusercontent.com')

def Preprocess_op():

    return dsl.ContainerOp(
        name='Preprocess Data ',
        image='abouzid/gcp-project-preprocess:latest',
        arguments=[],
        file_outputs={'preprocessed-dir': '/Preprocess/preprocess-data-dir.txt'}
    )
def Train_op(preprocess_data_dir : str):

    preprocess_data_dir
    return dsl.ContainerOp(
        name='Train Model ',
        image='abouzid/gcp-project-trainer:latest',
        arguments=['--preprocess-data-dir', preprocess_data_dir],
        file_outputs={'model-dir': '/trainer/model-dir.txt'}
    )
def Test_op(preprocess_data_dir : str , model_dir):
    return dsl.ContainerOp(
        name='Test Model ',
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
def Twitter_ML_Pipeline():
    _preprocess_op = Preprocess_op()
    _train_op = Train_op(
        _preprocess_op.outputs['preprocessed-dir']
    ).after(_preprocess_op)
    _test_op = Test_op(_preprocess_op.outputs['preprocessed-dir'],_train_op.outputs['model-dir']
    ).after(_train_op)

client.create_run_from_pipeline_func(Twitter_ML_Pipeline, arguments={})