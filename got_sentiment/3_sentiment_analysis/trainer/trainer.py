import os
import logging
import tempfile
import tensorflow as tf
import pandas as pd
import numpy as np
from explainable_ai_sdk.metadata.tf.v1 import KerasGraphMetadataBuilder
from preprocess import TextPreprocessor
from sklearn.model_selection import train_test_split
from google.cloud.storage import Client

sentiment_mapping={
    0:"negative",
    2:"neutral",
    4:"positive"
}

CLASSES = {'negative':0, 'positive': 1}  # label-to-int mapping
VOCAB_SIZE = 25000  # Limit on the number vocabulary size used for tokenization
MAX_SEQUENCE_LENGTH = 50  # Sentences will be truncated/padded to this length

def read_data_uri(uri):
    data_input = pd.read_csv(uri,encoding="latin1", header=None) \
        .rename(columns={
        0:"sentiment",
        1:"id",
        2:"time",
        3:"query",
        4:"username",
        5:"text"
    })[["sentiment","text"]]
    data_input["sentiment_label"] = data_input["sentiment"].map(sentiment_mapping)
    return data_input


def read_embbeded_data_uri(bucket,uri_data, temp,processor,EMBEDDING_DIM):
    client = Client()
    bucket = client.get_bucket(bucket)
    temp_folder = temp
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    blob = bucket.get_blob(uri_data)
    downloaded_file = blob.download_to_filename(temp_folder+'/glove.twitter.27B.50d.txt')
    def get_coaefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coaefs(*o.strip().split()) for o in
                            open(temp_folder+"/glove.twitter.27B.50d.txt","r",encoding="utf8"))
    word_index = processor._tokenizer.word_index
    nb_words = min(VOCAB_SIZE, len(word_index))
    embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= VOCAB_SIZE: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    return embedding_matrix




def preprocess_input (input):


    sents = input.text
    labels = np.array(input.sentiment_label.map(CLASSES))

    # Train and test split
    X, _, y, _ = train_test_split(sents, labels, test_size=0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # Create vocabulary from training corpus.
    processor = TextPreprocessor(VOCAB_SIZE, MAX_SEQUENCE_LENGTH)
    processor.fit(X_train)
    # Preprocess the data
    train_texts_vectorized = processor.transform(X_train)
    eval_texts_vectorized = processor.transform(X_test)


    return processor,y_train,y_test,train_texts_vectorized,eval_texts_vectorized

def create_model(vocab_size, embedding_dim, filters, kernel_sizes, dropout_rate, pool_size, embedding_matrix):

    # Input layer
    model_input = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

    # Embedding layer
    z = tf.keras.layers.Embedding(
        input_dim=vocab_size+1,
        output_dim=embedding_dim,
        input_length=MAX_SEQUENCE_LENGTH,
        weights=[embedding_matrix]
    )(model_input)

    z = tf.keras.layers.Dropout(dropout_rate)(z)

    # Convolutional block
    conv_blocks = []
    for kernel_size in kernel_sizes:
        conv = tf.keras.layers.Convolution1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="valid",
            activation="relu",
            bias_initializer='random_uniform',
            strides=1)(z)
        conv = tf.keras.layers.MaxPooling1D(pool_size=2)(conv)
        conv = tf.keras.layers.Flatten()(conv)
        conv_blocks.append(conv)

    z = tf.keras.layers.Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

    z = tf.keras.layers.Dropout(dropout_rate)(z)
    z = tf.keras.layers.Dense(100, activation="relu")(z)
    model_output = tf.keras.layers.Dense(1, activation="sigmoid")(z)

    model = tf.keras.models.Model(model_input, model_output)

    return model


def train_evaluate_explain_model(hparams):
    """Train, evaluate, explain TensorFlow Keras DNN Regressor.
    Args:
      hparams(dict): A dictionary containing model training arguments.
    Returns:
      history(tf.keras.callbacks.History): Keras callback that records training event history.
    """

    EMBEDDING_DIM=50
    POOL_SIZE=3
    KERNEL_SIZES=[2,5,8]


    input_data = read_data_uri(hparams['input-data-uri'])
    processor,y_train,y_test,train_vectorized,eval_vectorized = preprocess_input (input_data)
    embedding_matrix = read_embbeded_data_uri(hparams['bucket'] ,hparams['uri_data'] ,hparams['temp'],processor,EMBEDDING_DIM)



    model = create_model(VOCAB_SIZE, EMBEDDING_DIM, hparams['filters'], KERNEL_SIZES, hparams['dropout'],POOL_SIZE, embedding_matrix)
    logging.info(model.summary())
    # Compile model with learning parameters.
    optimizer = tf.keras.optimizers.Nadam(lr=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])


    #keras train
    history = model.fit(
        train_vectorized,
        y_train,
        epochs=hparams['n-checkpoints'],
        batch_size=hparams['batch-size'],
        validation_data=(eval_vectorized, y_test),
        verbose=2,
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_acc',
                min_delta=0.005,
                patience=3,
                factor=0.5),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                min_delta=0.005,
                patience=5,
                verbose=0,
                mode='auto'
            ),
            tf.keras.callbacks.History()
        ]
    )

    # Create a temp directory to save intermediate TF SavedModel prior to Explainable metadata creation.
    tmpdir = tempfile.mkdtemp()

    # Export Keras model in TensorFlow SavedModel format.
    tf.saved_model.save(model,(hparams['model-dir']))


    return history