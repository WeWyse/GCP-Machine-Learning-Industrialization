from kfp import dsl
import kfp
client = kfp.Client(host='https://1117fed2d4911d57-dot-europe-west1.pipelines.googleusercontent.com') # change


def Preprocess_op():
    return dsl.ContainerOp(
        name='Preprocess Data ',
        image='abouzid/gcp-project-preprocess:latest',
        arguments=["--input-data-uri",
                   "gs://rare-result-248415-tweet-sentiment-analysis/Data/sentiment_140/training_VB.csv"],
        file_outputs={'preprocessed-dir': '/Preprocess/preprocess-data-dir.txt'}
    )

def Train_op(preprocess_data_dir: str):
    preprocess_data_dir
    return dsl.ContainerOp(
        name='Train Model ',
        image='abouzid/gcp-project-trainer:latest',
        arguments=['--preprocess-data-dir', preprocess_data_dir],
        file_outputs={'model-dir': '/trainer/model-dir.txt'}
    )
def Test_op(preprocess_data_dir: str, model_dir):
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
def Use_case_train_VB():
    _preprocess_op = Preprocess_op()
    _train_op = Train_op(
        _preprocess_op.outputs['preprocessed-dir']
    ).after(_preprocess_op)
    _test_op = Test_op(_preprocess_op.outputs['preprocessed-dir'], _train_op.outputs['model-dir']
                       ).after(_train_op)
    _preprocess_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    _test_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    _train_op.execution_options.caching_strategy.max_cache_staleness = "P0D"


client.create_run_from_pipeline_func(Use_case_train_VB, arguments={})
