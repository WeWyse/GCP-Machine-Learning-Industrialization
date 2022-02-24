import os
import argparse

import trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Vertex custom container training args. These are set by Vertex AI during training but can also be overwritten.
    parser.add_argument('--model-dir', dest='model-dir',
                        default=os.environ['AIP_MODEL_DIR'], type=str, help='Model dir.')
    parser.add_argument('--bucket', dest='bucket',
                        default="'rare-result-248415-tweet-sentiment-analysis'", type=str, help='bucket name.')
    parser.add_argument('--input-data-uri', dest='input-data-uri',
                        default=os.environ['AIP_TRAINING_DATA_URI'], type=str, help='Training data GCS or BQ URI set during Vertex AI training.')
    parser.add_argument('--uri_data', dest='validation-data-uri',
                        default=os.environ['AIP_VALIDATION_DATA_URI'], type=str, help='embbeding data GCS or BQ URI set during Vertex AI training.')
    parser.add_argument('--temp-dir', dest='temp-dir',
                        default=os.environ['AIP_TEST_DATA_URI'], type=str, help='Temp dir set during Vertex AI training.')
    # Model training args.
    parser.add_argument('--dropout', dest='dropout', default=0.5, type=float, help='Float percentage of DNN nodes [0,1] to drop for regularization.')
    parser.add_argument('--filters', dest='filters', default=64, type=int, help='The dimensionality of the output space')
    parser.add_argument('--batch-size', dest='batch-size', default=128, type=int, help='Number of examples during each training iteration.')
    parser.add_argument('--n-checkpoints', dest='n-checkpoints', default=10, type=int, help='Number of model checkpoints to save during training.')

    args = parser.parse_args()
    hparams = args.__dict__

    trainer.train_evaluate_explain_model(hparams)