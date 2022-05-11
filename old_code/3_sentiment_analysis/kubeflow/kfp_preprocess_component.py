from kfp import dsl

def preprocess_op():

    return dsl.ContainerOp(
        name='Preprocess Data',
        image='abouzid /gcp-project:latest',
        arguments=[],
        file_outputs={
            'preprocessed-dir': '/app/x_train.npy',
        }
    )