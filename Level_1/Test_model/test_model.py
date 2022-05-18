import tensorflow as tf
import pandas as pd
import datetime
import numpy as np

def read_test_data(uri):
    y_test = pd.read_csv(uri + '/y_test.csv', encoding="latin1", header=None).to_numpy()
    x_test = pd.read_csv(uri + '/x_test.csv', encoding="latin1", header=None).to_numpy()

    return y_test, x_test


def evaluate_model(hparams):
    """Train, evaluate, explain TensorFlow Keras DNN Regressor.
    Args:
      hparams(dict): A dictionary containing model training arguments.
    Returns:
      history(tf.keras.callbacks.History): Keras callback that records training event history.
    """

    y_test, x_test = read_test_data(hparams['preprocess-data-dir'])
    model = tf.keras.models.load_model(hparams['model-dir'],custom_objects={'tf': tf})

    [loss, acc] = model.evaluate(x_test, y_test )
    scores = model.predict(x_test)
    predictions = np.array([int(np.round(i)) for i in scores ])
    confusion_matrix=tf.math.confusion_matrix(predictions, y_test)
    std_prediction = np.std(scores)

    if acc > hparams['performance-threshold']:
        nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        tf.saved_model.save(model, (hparams['model-validation-dir']) + '_' + nowTime)

    ouputfile = open("performance-model.txt", "w")
    ouputfile.write(str("loss : "+str(loss)+"/n acc :"+str(acc)+"/n matrix-co :"+str(confusion_matrix)
                        +"/n std-prediction :"+str(std_prediction)) )
