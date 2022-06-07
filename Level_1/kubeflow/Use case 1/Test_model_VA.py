from kfp import dsl
import kfp
client = kfp.Client(host='https://1117fed2d4911d57-dot-europe-west1.pipelines.googleusercontent.com')# change

def Preprocess_op():
    return dsl.ContainerOp(
        name='Preprocess Data ',
        image='abouzid/gcp-project-preprocess:latest',
        arguments=["--input-data-uri", "gs://rare-result-248415-tweet-sentiment-analysis/Data/sentiment_140/training_VA.csv"],
        file_outputs={'preprocessed-dir': '/Preprocess/preprocess-data-dir.txt'}
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
        })
@dsl.pipeline(
    name='Sentimental analyses Pipeline',
    description='An example pipeline.'
)
def Use_case_test_VB():
    _preprocess_op = Preprocess_op()
    _test_op = Test_op(_preprocess_op.outputs['preprocessed-dir'],"gs://rare-result-248415-tweet-sentiment-analysis/model/model-2022-06-03-08-20-49"
                       ).after(_preprocess_op)
    _preprocess_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    _test_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
client.create_run_from_pipeline_func(Use_case_test_VB, arguments={})