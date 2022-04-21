import os
import pandas as pd
import numpy as np
from preprocess import TextPreprocessor
import argparse
import logging
import json
import pickle

from google.cloud.storage import Client

CLASSES = {'negative': 0, 'positive': 1}  # label-to-int mapping
VOCAB_SIZE = 25000  # Limit on the number vocabulary size used for tokenization
MAX_SEQUENCE_LENGTH = 50  # Sentences will be truncated/padded to this length

sentiment_mapping = {
    0: "negative",
    2: "neutral",
    4: "positive"
}


def read_data_uri(uri):
    data_input = pd.read_csv(uri, encoding="latin1", header=None) \
        .rename(columns={
        0: "sentiment",
        1: "id",
        2: "time",
        3: "query",
        4: "username",
        5: "text"
    })[["sentiment", "text"]]
    data_input["sentiment_label"] = data_input["sentiment"].map(sentiment_mapping)
    return data_input


def read_embbeded_data_uri(bucket, uri_data, temp, processor, EMBEDDING_DIM):
    client = Client()
    bucket = client.get_bucket(bucket)
    temp_folder = temp
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    blob = bucket.get_blob(uri_data)
    downloaded_file = blob.download_to_filename(temp_folder + '/glove.twitter.27B.50d.txt')

    def get_coaefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coaefs(*o.strip().split()) for o in
                            open(temp_folder + "/glove.twitter.27B.50d.txt", "r", encoding="utf8"))
    word_index = processor._tokenizer.word_index
    nb_words = min(VOCAB_SIZE, len(word_index))
    embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= VOCAB_SIZE: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    return embedding_matrix


def preprocess_input(input, bucket, uri_data):
    sents = input.text
    labels = np.array(input.sentiment_label.map(CLASSES))

    # Train and test split

    processor = TextPreprocessor(VOCAB_SIZE, MAX_SEQUENCE_LENGTH)
    processor.fit(sents)
    # Preprocess the data
    train_texts_vectorized = processor.transform(sents)

    client = Client()
    bucket = client.get_bucket(bucket)
    logging.debug('Dumping processor pickle')
    blob = bucket.get_blob(uri_data)
    processor_dump = pickle.dumps(processor)
    blob.upload_from_string(processor_dump)

    return processor, train_texts_vectorized, labels


def run(hparams):
    logging.debug('preprocessing data Start')
    EMBEDDING_DIM = 50
    logging.debug('loading input data')
    input_data = read_data_uri(hparams['input-data-uri'])
    processor, vectorized_input, label = preprocess_input(input_data, hparams['bucket'], hparams['model-dir'])
    logging.debug('preprocessing input data Done')

    logging.debug('loading embedding data')
    embedding_matrix = read_embbeded_data_uri(hparams['bucket'], hparams['uri_data'], hparams['temp-dir'], processor,
                                              EMBEDDING_DIM)
    logging.debug('loading embedding done')

    input_data.to_csv(hparams['preprocess-data-dir'] + '/input.csv', index=False)
    logging.debug('saved preprocessed features data in ' + hparams['preprocess-data-dir'] + '/input.csv')

    pd.DataFrame(label).to_csv(hparams['preprocess-data-dir'] + '/label.csv', index=False, header=False)
    logging.debug('saved preprocessed label data in ' + hparams['preprocess-data-dir'] + '/label.csv')

    pd.DataFrame(vectorized_input).to_csv(hparams['preprocess-data-dir'] + '/vectorized_input.csv', index=False,
                                          header=False)
    logging.debug(
        'saved preprocessed and vectorized input data in ' + hparams['preprocess-data-dir'] + '/vectorized_input.csv')

    pd.DataFrame(embedding_matrix).to_csv(hparams['preprocess-data-dir'] + '/embedding_matrix.csv', index=False,
                                          header=False)
    logging.debug('saved embedding data in ' + hparams['preprocess-data-dir'] + '/embedding_matrix.csv')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Vertex custom container training args. These are set by Vertex AI during training but can also be overwritten.
    parser.add_argument('--model-dir', dest='model-dir',
                        default=os.environ['AIP_MODEL_DIR'], type=str, help='Model dir.')
    parser.add_argument('--preprocess-data-dir', dest='preprocess-data-dir',
                        default="", type=str, help="dirototory where to save preprocess data ")
    parser.add_argument('--input-data-uri', dest='input-data-uri',
                        default=os.environ['AIP_TRAINING_DATA_URI'], type=str,
                        help='Training data GCS or BQ URI set during Vertex AI training.')
    parser.add_argument('--uri_data', dest='validation-data-uri',
                        default=os.environ['AIP_VALIDATION_DATA_URI'], type=str,
                        help='embedding data GCS or BQ URI set during Vertex AI training.')
    parser.add_argument('--temp-dir', dest='temp-dir',
                        default=os.environ['AIP_TEST_DATA_URI'], type=str,
                        help='Temp dir set during Vertex AI training.')

    parser.add_argument('--load_json', dest='-load_json', default='config.json',
                        help='Load settings from file in json format. Command line options override values in file.')

    args = parser.parse_args()
    if args.load_json:
        with open(args.load_json, 'rt') as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            args = parser.parse_args(namespace=t_args)

    hparams = args.__dict__
    run(hparams)
